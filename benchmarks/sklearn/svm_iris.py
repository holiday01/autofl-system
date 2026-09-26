"""
SVM classifier on the Iris dataset (sklearn).

Trains a Support Vector Machine with an RBF kernel to classify the three
Iris species.  No external data files needed — loads via sklearn.datasets.
"""
import argparse
from sklearn import svm, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser(description="SVM on Iris (sklearn)")
    parser.add_argument("--C", type=float, default=1.0, help="SVM regularisation (default: 1.0)")
    parser.add_argument("--kernel", type=str, default="rbf", help="SVM kernel (default: rbf)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = svm.SVC(C=args.C, kernel=args.kernel, random_state=args.seed, probability=True)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))


if __name__ == "__main__":
    main()
