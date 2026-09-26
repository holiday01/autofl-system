from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from transformers.trainer_pt_utils import get_parameter_names

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def _load_raw_datasets(config: dict):
    task_name = config.get("task_name")
    if task_name is not None:
        return load_dataset("nyu-mll/glue", task_name)
    data_files = {}
    train_file = config.get("train_file")
    validation_file = config.get("validation_file")
    if train_file:
        data_files["train"] = train_file
    if validation_file:
        data_files["validation"] = validation_file
    extension = (train_file or validation_file).split(".")[-1]
    return load_dataset(extension, data_files=data_files)


def _get_label_info(raw_datasets: dict, config: dict):
    task_name = config.get("task_name")
    if task_name is not None:
        is_regression = task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            label_list = []
            num_labels = 1
    else:
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            label_list = []
            num_labels = 1
        else:
            label_list = sorted(raw_datasets["train"].unique("label"))
            num_labels = len(label_list)
    return num_labels, label_list, is_regression


def build_model(config: dict):
    """
    config keys:
        model_name_or_path (str, required)
        task_name (str, optional)
        train_file / validation_file (str, optional — used to infer num_labels when task_name is None)
        use_slow_tokenizer (bool, default False)
        ignore_mismatched_sizes (bool, default False)
        trust_remote_code (bool, default False)
    Returns: (model, tokenizer)
    """
    model_name_or_path = config["model_name_or_path"]
    trust_remote_code = config.get("trust_remote_code", False)

    raw_datasets = _load_raw_datasets(config)
    num_labels, label_list, is_regression = _get_label_info(raw_datasets, config)

    hf_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=config.get("task_name"),
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=not config.get("use_slow_tokenizer", False),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=hf_config,
        ignore_mismatched_sizes=config.get("ignore_mismatched_sizes", False),
        trust_remote_code=trust_remote_code,
    )

    # Align label mappings
    task_name = config.get("task_name")
    label_to_id = None
    from transformers import PreTrainedConfig as _PTC
    if (
        model.config.label2id != _PTC(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id_: label for label, id_ in hf_config.label2id.items()}
    elif task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id_: label for label, id_ in model.config.label2id.items()}

    return model, tokenizer


def build_dataloader(config: dict, split: str) -> DataLoader:
    """
    config keys (beyond build_model keys):
        per_device_train_batch_size (int, default 8)
        per_device_eval_batch_size (int, default 8)
        max_length (int, default 128)
        pad_to_max_length (bool, default False)
    split: "train" or "validation"
    """
    model_name_or_path = config["model_name_or_path"]
    trust_remote_code = config.get("trust_remote_code", False)
    task_name = config.get("task_name")
    max_length = config.get("max_length", 128)
    pad_to_max_length = config.get("pad_to_max_length", False)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=not config.get("use_slow_tokenizer", False),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_datasets = _load_raw_datasets(config)
    num_labels, label_list, is_regression = _get_label_info(raw_datasets, config)

    if task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[task_name]
    else:
        non_label_cols = [n for n in raw_datasets["train"].column_names if n != "label"]
        if "sentence1" in non_label_cols and "sentence2" in non_label_cols:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif len(non_label_cols) >= 2:
            sentence1_key, sentence2_key = non_label_cols[0], non_label_cols[1]
        else:
            sentence1_key, sentence2_key = non_label_cols[0], None

    # Rebuild label_to_id to be consistent with build_model
    from transformers import AutoConfig as _AC, PreTrainedConfig as _PTC
    hf_config = _AC.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        trust_remote_code=trust_remote_code,
    )
    label_to_id = None
    if (
        hf_config.label2id != _PTC(num_labels=num_labels).label2id
        and task_name is not None
        and not is_regression
    ):
        label_name_to_id = {k.lower(): v for k, v in hf_config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
    elif task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if pad_to_max_length else False

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=padding, max_length=max_length, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                result["labels"] = [label_to_id[l] for l in examples["label"]]
            else:
                result["labels"] = examples["label"]
        return result

    processed = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc=f"Tokenizing {split}",
    )

    if split == "train":
        dataset = processed["train"]
        batch_size = config.get("per_device_train_batch_size", 8)
        shuffle = True
    else:
        eval_key = "validation_matched" if task_name == "mnli" else "validation"
        dataset = processed[eval_key]
        batch_size = config.get("per_device_eval_batch_size", 8)
        shuffle = False

    if pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    return DataLoader(dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)


def train_step(
    model: nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    config: dict,
) -> torch.Tensor:
    """
    Performs a single forward-backward-update step.
    config keys used:
        gradient_accumulation_steps (int, default 1)
    Returns the raw (unscaled) loss for this batch.
    """
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

    model.train()
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    outputs = model(**batch)
    loss = outputs.loss

    scaled_loss = loss / gradient_accumulation_steps
    scaled_loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    return loss.detach()