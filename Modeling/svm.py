from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from scipy import sparse

# Load datasets
X_train_sparse = sparse.load_npz("X_train.npz")
X_test_sparse = sparse.load_npz("X_test.npz")
y_train = pd.read_csv("y_train.csv", index_col=0, squeeze=True)
y_test = pd.read_csv("y_test.csv", index_col=0, squeeze=True)

# Convert sparse matrices to dense, if needed
X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

# Fit model
svm_linear = LinearSVC(max_iter=10000)  # increase max_iter if convergence warning
svm_linear.fit(X_train, y_train)

# Evaluate model
y_pred = svm_linear.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
