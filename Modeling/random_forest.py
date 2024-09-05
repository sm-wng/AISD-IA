from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
import pandas as pd
    


X_train_sparse = sparse.load_npz("X_train.npz")
X_test_sparse = sparse.load_npz("X_test.npz")
y_train = pd.read_csv("y_train.csv", index_col=0, squeeze=True)
y_test = pd.read_csv("y_test.csv", index_col=0, squeeze=True)
features = np.load("features.npy", allow_pickle=True)

# Convert sparse matrices to dense matrices
X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

# Train rf model
rf = RandomForestClassifier(criterion='entropy', class_weight='balanced')
rf.fit(X_train, y_train)

# Test prediction with a classification report
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot important features
idx = np.argsort(rf.feature_importances_)[::-1] # Important features index
val = np.sort(rf.feature_importances_)[::-1] # Feature importance score
fts = [features[i] for i in idx] # Top features

# plt.barh(fts[:10], val[:10])
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.title("Important Features of RF Classifier")

# Learning Curve
size = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
estimator = RandomForestClassifier(criterion='entropy', class_weight='balanced')
train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, train_sizes=size)
plt.plot(train_sizes,np.mean(train_scores,axis=1), label='Train')
plt.plot(train_sizes,np.mean(test_scores,axis=1), label='Test')
plt.title("Leaning Curve of Random Forest Model")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()