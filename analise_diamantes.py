import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1️ Carregar o conjunto de dados

print("🔹 Carregando dados...")
df = sns.load_dataset('diamonds')  # dataset embutido no Seaborn

print("Dados carregados com sucesso!")
print(df.head())


# 2️ Informações gerais

print("\n📊 Informações gerais:")
print(df.info())

print("\n📈 Estatísticas descritivas:")
print(df.describe())


# 3️ Análise univariada

plt.figure(figsize=(8,5))
sns.histplot(df['price'], kde=True, bins=50)
plt.title('Distribuição dos preços dos diamantes')
plt.xlabel('Preço (USD)')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['carat'], kde=True, bins=50, color='orange')
plt.title('Distribuição do peso (carat)')
plt.xlabel('Peso (carat)')
plt.ylabel('Frequência')
plt.show()


# 4️ Análise categórica

plt.figure(figsize=(8,5))
sns.boxplot(x='cut', y='price', data=df, palette='viridis')
plt.title('Preço dos diamantes por tipo de corte')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='color', y='price', data=df, palette='coolwarm')
plt.title('Preço dos diamantes por cor')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='clarity', y='price', data=df, palette='magma')
plt.title('Preço dos diamantes por clareza')
plt.show()

# 5️ Correlações numéricas

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='crest')
plt.title('Mapa de calor das correlações numéricas')
plt.show()

# 6️ Relações entre variáveis

plt.figure(figsize=(8,6))
sns.scatterplot(x='carat', y='price', data=df, alpha=0.5)
plt.title('Relação entre peso (carat) e preço')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='depth', y='price', data=df, alpha=0.5)
plt.title('Relação entre profundidade e preço')
plt.show()

print("\n✅ Análise concluída com sucesso!")
