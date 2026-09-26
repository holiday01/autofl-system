"""
XGBoost binary classifier on the breast cancer dataset.

Trains an XGBoost gradient-boosted tree ensemble to distinguish malignant
from benign breast tumours.  No external data files needed — loads via
sklearn.datasets.
"""
import argparse
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser(description="XGBoost on breast_cancer (sklearn dataset)")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of boosting rounds (default: 200)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Maximum tree depth (default: 4)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate / eta (default: 0.1)")
    parser.add_argument("--subsample", type=float, default=0.8,
                        help="Row sub-sample ratio (default: 0.8)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target  # (569, 30), binary labels

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features (optional but consistent with sklearn scripts)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        subsample=args.subsample,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=args.seed,
        verbosity=1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=10,
    )

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f"\nTest accuracy : {acc:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()
