from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

df = load_iris()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(predictions)

d = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
print(list(map(lambda x: d[x], predictions)))

confusion_matrix(y_test, predictions)