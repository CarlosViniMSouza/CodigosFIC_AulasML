import numpy as np

p = 0.6
nsim = 10
nhead = 0
exit = []

for i in range(0, nsim):
  
  if(np.random.uniform() < p):
    nhead = nhead + 1
    exit.append(1)
  else:
    exit.append(0)

print("Saida: ", exit)
print("Frequencia de caras: ", nhead/nsim)