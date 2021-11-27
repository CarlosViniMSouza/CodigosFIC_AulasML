import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn import linear_model

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
plt.show()