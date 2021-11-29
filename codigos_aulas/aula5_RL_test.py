from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(solver = 'lbfgs', multi_class = 'auto', max_iter = 1000).fit(X, y)

print("\nPrediction: ", clf.predict(X[:2, :]))
print("\nArray: ", clf.predict_proba(X[:2, :]))
print("\nAccuracy: ", clf.score(X, y))
