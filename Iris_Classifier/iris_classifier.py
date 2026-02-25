import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model with Regularization (max_depth)
# We limit the depth to prevent the model from becoming too complex/overfitting
clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluation
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, clf.predict(X_test))

# 5. Cross-Validation (Better metric for small datasets)
cv_scores = cross_val_score(clf, X, y, cv=5)

print("--- Iris Classification Results ---")
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {test_acc * 100:.2f}%")
print(f"Mean Cross-Validation Score: {cv_scores.mean() * 100:.2f}%")
print("\nDetailed Report (Test Set):")
print(classification_report(y_test, clf.predict(X_test), target_names=iris.target_names))