import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/info.csv')
print(df.head())

df['A1'] = df['A1'].replace(['escuro', 'claro'], ['0', '1'])
df['A1'] = df['A1'].astype(int)

df['A2'] = df['A2'].replace(['baixo', 'alto'], ['0', '1'])
df['A2'] = df['A2'].astype(int)

df['Classe'] = df['Classe'].replace(['-', '+'], ['0', '1'])
df['Classe'] = df['Classe'].astype(int)

print(type(df['Classe']))

X = df.iloc[:, 0:-1]
y = df.iloc[:, ['Classe']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(predictions)

# Aplicando a R.L.:
y_pred_log_reg = classifier.predict(X_test)
acc_log_reg = round(classifier.score(X_train, y_train) * 100, 2)
print ("Acuracia: " + str(acc_log_reg) + ' %')
