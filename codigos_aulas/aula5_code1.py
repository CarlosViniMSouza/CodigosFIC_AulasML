import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = np.arange(12).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1])
plt.plot(X, y, 'ko')

# Ajustando o Modelo de Regressao Logistica:
model = linear_model.LogisticRegression(C=1e5)
model.fit(X, y)
loss = expit(X * model.coef_ + model.intercept_).ravel()
plt.plot(X, loss, color='red', linewidth=3)

# Ajustando o Model de Regressao Linear:
linear = linear_model.LinearRegression()
linear.fit(X, y)
plt.plot(X, linear.coef_ * X + linear.intercept_, linewidth=2, linestyle='dashed')
plt.axhline(.5, color='.5')
plt.savefig('logistic.eps')
plt.show() # --> vai mostrar o grafico.

p = model.predict_proba(X)
print(p)

print("Acuracia: ", round(model.score(X, y), 2))


# Modelo de Regressao Logistica usa a funcao logistica:
x = np.linspace(-10, 10, 100)
z = 1/(1 + np.exp(-x))

plt.figure(figsize=(6, 4))
plt.plot(x, z)
plt.xlabel("x", fontsize=15)
plt.ylabel("h(x)", fontsize=15)
plt.savefig('logistic-function.eps')
plt.show() # --> vai mostrar o grafico.

# Classificacao de Dados:
"""
random.seed(42)
data = pd.read_csv('', header=(0))
data = data.dropna(axis='')

classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)

print("Num. Linhas & Colunas da matriz de atributos: ", data.shape)
attributes = list(data.columns)
data.head(10)
"""

# Convertendo para o formato Numpy:
"""
data = data.to_numpy()
nrow, ncol = data.shape
y = data[:, -1]
X = data[:, 0:ncol - 1]

scaler = StandardScaler().fit(X)
X = scaler.transform(X)
"""

# Selecionamos os conjuntos de teste e treino:
p = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=p, random_state=4)

# Classificando por Regressao Logistica:
model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Acuracia: ", round(model.score(x_test, y_test), 2))