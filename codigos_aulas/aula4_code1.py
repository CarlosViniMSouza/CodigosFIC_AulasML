## Classificador Naive Bayes:
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

data = pd.read_csv('data/Iris.csv', header=0)
data = data.dropna(axis='rows')

classes = np.array(pd.unique(data[data.columns[-1]]), dtype=str)

print("Num. Linhas & Colunas da Matriz de Atributos: ", data.shape)
attributes = list(data.columns)

print(data.head(10))

data = data.to_numpy()
nrow, ncol = data.shape
y = data[:, -1]
X = data[:, 0:ncol - 1]

## Selecionando os conjuntos de Treino e Teste:
from sklearn.model_selection import train_test_split
p = 0.7
X_train, x_test, y_train, y_test = train_test_split(X, y, train_size = p)

## Classificacao: implementacao do Metodo:
def likelyhood(y, Z):

  def gaussian(x, mu, sig):
    p = (1/np.sqrt(2 * np.pi * sig)) * np.exp((-1/2) * ((x - mu)/sig)**2)
    return p

  lk = 1
  for j in np.arange(0, Z.shape[1]):
    m = np.mean(Z[:, j])
    s = np.std(Z[:, j])
    lk = lk * gaussian(y[j], m, s)
  return lk

## Estimacao de cada classe:
P = pd.DataFrame(data=np.zeros((X_test.shape[0], len(classes))), columns = classes)

for i in np.arange(0, len(classes)):
  elements = tuple(np.where(y_train == classes[i]))
  Z = X_train[elements,:][0]
  for j in np.arange(0,X_test.shape[0]):
    x = X_test[j,:]
    pj = likelyhood(x,Z)
    P[classes[i]][j] = pj*len(elements)/X_train.shape[0]

# Para as observações no conjunto de teste, a probabilidade pertencer a cada classe:
P.head(10)

from sklearn.metrics import accuracy_score

y_pred = []
for i in np.arange(0, P.shape[0]):
  c = np.argmax(np.array(P.iloc[[i]]))
  y_pred.append(P.columns[c])

y_pred = np.array(y_pred, dtype=str)
score = accuracy_score(y_pred, y_test)
print('Accuracy:', score)
