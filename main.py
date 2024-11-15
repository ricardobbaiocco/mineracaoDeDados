import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para o arquivo de dados
DATA_FILE = 'data/Titanic-Dataset.csv'
data = pd.read_csv(DATA_FILE)

# Limpeza e preparação dos dados
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Codificar variáveis categóricas
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)  #
data['Embarked'] = data['Embarked'].apply(lambda x: 1 if x == 'S' else 0)

# Variáveis independentes e dependente
X = data[['Pclass', 'Age', 'SibSp', 'Sex_male', 'Embarked']]
y = data['Survived']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#CLASSIFICAÇÃO
# Classificação com Floresta Aleatória
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
predictions = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do modelo de classificação: {accuracy:.2f}')

# Visualização da distribuição de sobreviventes
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=data, palette='viridis')
plt.title('Distribuição de Sobreviventes')
plt.xlabel('Sobrevivente (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.show()

# AGRUPAMENTO
# Normalizar os dados para o agrupamento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar os clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Cluster', data=data, palette='viridis')
plt.title('Clusters de Passageiros com K-Means')
plt.xlabel('Idade')
plt.ylabel('Tarifa')
plt.legend(title='Cluster')
plt.show()

#REGRESSÃO
# Prever a tarifa  usando outras variáveis
X_reg = data[['Pclass', 'Age', 'SibSp', 'Sex_male', 'Embarked']]
y_reg = data['Fare']

# Dividir os dados em treino e teste
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Cria e treina o modelo de regressão
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Prever tarifas
y_pred_reg = reg_model.predict(X_test_reg)

# Visualizar a relação entre valores reais e previstos
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test_reg, y=y_pred_reg)
plt.title('Regressão Linear: Tarifas Reais vs. Previsões')
plt.xlabel('Tarifas Reais')
plt.ylabel('Tarifas Previstos')
plt.plot([0, 500], [0, 500], '--', color='red', linewidth=2)
plt.show()
