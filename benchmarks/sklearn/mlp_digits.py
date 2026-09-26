"""
MLP classifier on the Digits dataset (sklearn.neural_network.MLPClassifier).

Trains a two-hidden-layer MLP to recognise hand-written digits (0-9).
No external data files needed — loads via sklearn.datasets.
"""
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def main():
    parser = argparse.ArgumentParser(description="MLP on Digits (sklearn)")
    parser.add_argument("--hidden-layer-sizes", type=int, nargs="+", default=[128, 64],
                        help="Sizes of hidden layers (default: 128 64)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (default: 1e-3)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Max training iterations (default: 200)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Test fraction (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Load data
    digits = load_digits()
    X, y = digits.data, digits.target  # (1797, 64), labels 0-9

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # Normalise features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(args.hidden_layer_sizes),
        learning_rate_init=args.lr,
        max_iter=args.epochs,
        random_state=args.seed,
        verbose=True,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(metrics.classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
