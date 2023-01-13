#importando o pandas e utilizando para ler a base de dados 
import pandas as pd

tabela = pd.read_csv("advertising.csv")
print(tabela)

#biblioteca dos graficos
import seaborn as sns
import matplotlib.pyplot as plt

#grafico de calor para melhor vizualisação da correlação
sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

#treinando a IA
from sklearn.model_selection import train_test_split

y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)

#criação das  inteligencias aritificiais
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

from sklearn import metrics

# criando as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparando as previsões das duas inteligencias
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))  

# comparando as previsões das duas inteligencias de forma grafica
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()

#nova previsão
nova_tabela = pd.read_csv("novos.csv")
print(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)