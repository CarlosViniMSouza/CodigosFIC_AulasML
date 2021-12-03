import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

var = pd.read_csv('Codigos\data\diabetes\diabetes.csv')
print(var.head())

# Procuradno informacoes sobre cada coluna:
print(var.describe())

# Alterando a minima da tabela 'BloodPressure':
var.loc[var['BloodPressure'] == 0, ['BloodPressure']] = var['BloodPressure'].mean()

# Outra sugestao:
var.loc[var['BloodPressure'] == 0, ['BloodPressure']] = np.NAN

# Checando se a alteracao foi bem sucedida:
print(var.describe())

# Substituindo NaN pela media:
var['BloodPressure'].fillna(var['BloodPressure'].mean(), inplace = True) # Deu problema com essa linha!

# sns.heatmap(var.corr(), xticklabels=var.columns, yticklabes=var.columns, annot=True) --> mapeando correlacoes de variaveis

X = var.iloc[:, 0:-1]
y = var.loc[:, ['Outcome']]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logVar = LogisticRegression(solver = 'lbfgs', multi_class='auto', max_iter=1000)
logVar.fit(X_train, Y_train)

yPred = logVar.predict(X_test)

print("Acuracia: ", accuracy_score(yPred, Y_test))
