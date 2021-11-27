import numpy as np
import matplotlib.pyplot as plt

p = 0.6
vp = []
vsim = []
Nmax = 2000

for nsim in np.arange(1, Nmax, 2):
  
  nhead = 0
  for i in range(1, nsim):
    
    if(np.random.uniform() < p):
      nhead = nhead + 1
  vp.append(nhead/nsim)
  vsim.append(nsim)

# Veja os resultados:
plt.figure(figsize=(8, 6))

plt.plot(vsim, vp, linestyle = '-', color = "gray", linewidth = 2, label = 'Valor Simulado')

plt.axhline(y = p, color = 'g', linestyle = '--', label = 'Valor Teorico')

plt.ylabel("Fracao de Caras", fontsize = 20)
plt.xlabel("Numero de Experimentos", fontsize = 20)

plt.xlim([0.0, Nmax])
plt.ylim([0.0, 1.0])

plt.legend()
plt.show()

"""
Erro(s) no codigo original:

1 - Na ultima linha, eh exigido 2 parametros.

  Solucao: Para funcionar, basta remover o parametro 'True'.
"""