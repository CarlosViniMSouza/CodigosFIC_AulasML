import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/titanic/train.csv')
df.head(10)

X = df.iloc[:, 2:]
y = df.iloc[:, 1]

X = X.drop(['Name', 'Cabin', 'Ticket', 'Embarked'], axis = 1)
X.Sex = LabelEncoder().fit_transform(X.Sex)
X = X.fillna(X.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(predictions)