import numpy as np
import pandas as pd
from numpy import log2 as log

eps = np.finfo(float).eps
var = pd.read_csv("data/titanic/train.csv")

var = var.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
var = var.dropna()
var['Age'] = var['Age'].interpolate()

entropia = 0
values = var['Survived'].unique()

for value in values:
    p = var['Survived'].value_counts()[value]/len(var.Survived)
    entropia = entropia + (-(p*log(p)))
# entropia de todo DB:
print("Valor da entropia: ", entropia)

def tributo_entropia(col):
    y = var['Survived'].unique()
    x = var[col].unique()
    entropia = 0
    for i in x:
        e = 0
        for j in y:
            num = len(var[col][var[col] == i][var['Survived'] == j])
            den = len(var[col][var[col] == i])
            p = num/(den+eps)
            e = e + (-(p*log(p+eps)))
        p2 = den/len(var)
        entropia = entropia + (-(p2*e))
    return abs(entropia)

# entropia tendo por base a idade:
print("Valor da entropia do Atributo (Age): ", tributo_entropia('Age'))

# entropia tendo por base o sexo:
print("Valor da entropia do Atributo (Sex): ", tributo_entropia('Sex'))