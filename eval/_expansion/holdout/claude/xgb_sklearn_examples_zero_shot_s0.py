import numpy as np
import xgboost as xgb
from sklearn.datasets import fetch_california_housing, load_digits, load_iris, make_regression
from sklearn.model_selection import train_test_split
from urllib.error import HTTPError


def build_model(config):
    task = config.get("task", "classification")
    params = config.get("model_params", {})

    if task == "regression":
        model = xgb.XGBRegressor(n_jobs=1, **params)
    elif task == "multiclass":
        model = xgb.XGBClassifier(n_jobs=1, objective="multi:softmax", **params)
    else:
        model = xgb.XGBClassifier(n_jobs=1, **params)

    return model


def build_dataloader(config, split):
    dataset = config.get("dataset", "digits")
    test_size = config.get("test_size", 0.25)
    random_state = config.get("random_state", 0)

    if dataset == "digits":
        n_class = config.get("n_class", 2)
        data = load_digits(n_class=n_class)
        X, y = data["data"], data["target"]
    elif dataset == "iris":
        data = load_iris()
        X, y = data["data"], data["target"]
    elif dataset == "california_housing":
        try:
            X, y = fetch_california_housing(return_X_y=True)
        except HTTPError:
            X, y = make_regression(n_samples=20640, n_features=8, random_state=1234)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if split == "train":
        return {"X": X_train, "y": y_train}
    elif split == "test":
        return {"X": X_test, "y": y_test}
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train' or 'test'.")


def train_step(model, batch, optimizer, config):
    X, y = batch["X"], batch["y"]

    early_stopping_rounds = config.get("early_stopping_rounds", None)
    eval_metric = config.get("eval_metric", None)
    eval_set = config.get("eval_set", None)

    fit_kwargs = {}
    if early_stopping_rounds is not None:
        model.set_params(early_stopping_rounds=early_stopping_rounds)
    if eval_metric is not None:
        model.set_params(eval_metric=eval_metric)
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set

    model.fit(X, y, **fit_kwargs)

    predictions = model.predict(X)

    task = config.get("task", "classification")
    if task == "regression":
        from sklearn.metrics import mean_squared_error
        loss = mean_squared_error(y, predictions)
    else:
        loss = float(np.mean(predictions != y))

    return {"model": model, "loss": loss, "predictions": predictions}