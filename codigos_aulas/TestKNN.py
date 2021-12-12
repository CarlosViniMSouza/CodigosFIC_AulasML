import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data/Iris.csv')

df.head(10)

X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(predictions)

pd.DataFrame({'True Label': y_test, 
  'Predicted Label': predictions, 
  'Correct': y_test == predictions})

confusion_matrix(y_test, predictions)