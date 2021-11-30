## TITANIC - REGRESSÃO LOGISTICA:

# Importando Libs:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# Carregando Tabelas: 
train = pd.read_csv('data/titanic/train.csv')
test = pd.read_csv('data/titanic/test.csv')


# Vendo as tabelas:
print(train.head())
print(test.head())


## Tabela de TRAIN:
# Convertendo o Atributo $Sex$ de categórico para numérico:
train['Sex'] = train['Sex'].replace(['male','female'],['0','1'])
train['Sex'] = train['Sex'].astype(int)

# Substituindo NULL pela mediana das idades:
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Sex'] = train['Sex'].astype(int)

# Substituindo NULL pela mediana dos precos:
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['Fare'] = train['Fare'].astype(int)


## Tabela de TEST:
# Convertendo o Atributo $Sex$ de categórico para numérico:
test['Sex'] = test['Sex'].replace(['male','female'],['0','1'])
test['Sex'] = test['Sex'].astype(int)

# Substituindo NULL pela mediana das idades:
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Sex'] = test['Sex'].astype(int)

# Substituindo NULL pela mediana dos precos:
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
test['Fare'] = test['Fare'].astype(int)


# Selecionando colunas a serem removidas:
train.columns
test.columns
colunas_eliminar = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
train = train.drop(['PassengerId'], axis=1)
train = train.drop(colunas_eliminar, axis=1)
test = test.drop(colunas_eliminar, axis=1)


# Definir conjuntos de treino e teste:
X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test = test.drop("PassengerId", axis=1).copy()


# Aplicando a R.L.:
clf = LogisticRegression()  # Inicializa o classificador
clf.fit(X_train, y_train)   # Treina o classificador
y_pred_log_reg = clf.predict(X_test)
acc_log_reg = round( clf.score(X_train, y_train) * 100, 2)
print ("Acuracia: " + str(acc_log_reg) + ' %')