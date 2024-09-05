import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy import sparse

X_train_sparse = sparse.load_npz("X_train.npz")
X_test_sparse = sparse.load_npz("X_test.npz")
y_train = pd.read_csv("y_train.csv", index_col=0, squeeze=True)
y_test = pd.read_csv("y_test.csv", index_col=0, squeeze=True)

# Convert sparse matrices to dense matrices
X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

model = make_pipeline(StandardScaler(), BernoulliNB())

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
