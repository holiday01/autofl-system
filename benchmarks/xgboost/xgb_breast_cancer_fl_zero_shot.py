import numpy as np
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def build_model(config):
    return xgb.XGBClassifier(
        n_estimators=config.get("rounds_per_step", 1),
        max_depth=config.get("max_depth", 4),
        learning_rate=config.get("lr", 0.1),
        subsample=config.get("subsample", 0.8),
        eval_metric="logloss",
        random_state=config.get("seed", 42),
        verbosity=0,
    )


def build_dataloader(config, split):
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.get("test_size", 0.2),
        random_state=config.get("seed", 42),
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if split == "train":
        return [(X_train, y_train)]
    if split == "test":
        return [(X_test, y_test)]
    raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'test'.")


def train_step(model, batch, optimizer, config):
    # optimizer is accepted for API compatibility but unused — XGBoost
    # manages its own optimisation internally.
    X, y = batch

    # Warm-start from the accumulated booster if the model has been fit before;
    # each call adds rounds_per_step new trees on top of the existing ensemble.
    prior_booster = model.get_booster() if hasattr(model, "n_features_in_") else None
    model.set_params(n_estimators=config.get("rounds_per_step", 1))
    model.fit(X, y, xgb_model=prior_booster)

    loss = metrics.log_loss(y, model.predict_proba(X)[:, 1])
    return {"loss": float(loss), "num_examples": int(len(y))}