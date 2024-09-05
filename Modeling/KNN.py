from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

X_train_sparse = sparse.load_npz("X_train.npz")
X_test_sparse = sparse.load_npz("X_test.npz")
y_train = pd.read_csv("y_train.csv", index_col=0, squeeze=True)
y_test = pd.read_csv("y_test.csv", index_col=0, squeeze=True)

# Convert sparse matrices to dense matrices
X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

# k = 3
# for k in range(3, 10):
#     knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
#     knn.fit(X_train, y_train)

#     y_pred = knn.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))
    
# Learning Curve
size = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
estimator = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, train_sizes=size)
plt.plot(train_sizes,np.mean(train_scores,axis=1), label='Train')
plt.plot(train_sizes,np.mean(test_scores,axis=1), label='Test')
plt.title("Leaning Curve of KNN")
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

