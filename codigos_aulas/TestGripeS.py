import math
"""
Tabela da Aula em formato de Dicionario:

1 - SIM ="s
2 - NAO = "n"
3 - MEDIA = m
4 - FORTE = f

Onde:
Calafrio"s" = "N"ao
Coriza = Nao
Cefaleia = Forte
Febre = SIM
"""

tabela = {
  "Calafrios": ["s", "s", "s", "n", "n", "n", "n", "s"],
  "Cefaleia": ["m", "n", "f", "m", "n", "f", "f", "m"],
  "Coriza": ["n", "s", "n", "s", "n", "s", "s", "s"],
  "Febre": ["s", "n", "s", "s", "n", "s", "n", "s"],
  "GRIPE": ["n", "s", "s", "s", "n", "s", "n", "s"]
}

"""
Formula: print((-4/6)*math.log(4/6, 2)-(2/6)*math.log(2/6, 2))
"""

# Separando a coluna gripe entre "SIM" e "NAO":

# Contadores:
"""
quantGripeSim = 0
quantGripeNao = 0

for i in range(len(tabela['GRIPE'])):

  if(tabela['GRIPE'][i] == "s"):
    quantGripeSim += 1
  else:
    quantGripeNao += 1
"""

# Agora vamos dividir cada coluna nas respectivas respostas:

# Laco FOR para a coluna "Calafrios"
# Contadores:
quantCalaSim = 0
quantCalaNao = 0

for i in range(0, len(tabela['Calafrios'])):

  if (tabela['Calafrios'][i] == "s" and tabela["GRIPE"][i] == "s"):
    quantCalaSim += 1
  else:
    quantCalaNao += 1


# Laco FOR para a coluna "Cefaleia"
# Contadores:
quantCefaSim = 0
quantCefaNao = 0

for i in range(0, len(tabela['Cefaleia'])):

  if (tabela['Cefaleia'][i] == "m" and tabela["GRIPE"][i] == "s"):
    quantCefaSim += 1
  else:
    quantCefaNao += 1

# Laco FOR para a coluna "Coriza"
# Contadores:
quantCoriSim = 0
quantCoriNao = 0

for i in range(0, len(tabela['Coriza'])):

  if (tabela['Coriza'][i] == "s" and tabela["GRIPE"][i] == "s"):
    quantCoriSim += 1
  else:
    quantCoriNao += 1


# Laco FOR para a coluna "Febre"
# Contadores:
quantFebreSim = 0
quantFebreNao = 0

for i in range(0, len(tabela['Febre'])):

  if (tabela['Febre'][i] == "s" and tabela["GRIPE"][i] == "s"):
    quantFebreSim += 1
  else:
    quantFebreNao += 1


# Aplicando a Formula
"""
EntroGripe = ((-1*quantGripeSim/6) * math.log(quantGripeSim/6, 2) - (quantGripeNao/6) * math.log(quantGripeNao/6, 2))
print("O resultado de Entropia(GRIPE) eh: ", EntroGripe)
"""

EntroCala = ((-1*quantCalaSim/6) * math.log(quantCalaSim/6, 2) - (quantCalaNao/6) * math.log(quantCalaNao/6, 2))
print("O resultado de Entropia(Calafrios) eh: ", EntroCala)

EntroCefa = ((-1*quantCefaSim/6) * math.log(quantCefaSim/6, 2) - (quantCefaNao/6) * math.log(quantCefaNao/6, 2))
print("O resultado de Entropia(Cefaleia) eh: ", EntroCefa)

EntroCori = ((-1*quantCoriSim/6) * math.log(quantCoriSim/6, 2) - (quantCoriNao/6) * math.log(quantCoriNao/6, 2))
print("O resultado de Entropia(Cefaleia) eh: ", EntroCori)

EntroFebre = ((-1*quantFebreSim/6) * math.log(quantFebreSim/6, 2) - (quantFebreNao/6) * math.log(quantFebreNao/6, 2))
print("O resultado de Entropia(Cefaleia) eh: ", EntroFebre)