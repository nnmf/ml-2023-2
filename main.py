import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA



# Leio o arquivo no formato CSV
data = pd.read_csv('card_transdata.csv')


#print(data)

X = data.drop('fraud', axis=1)
y = data['fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalização dos dados para o KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Treinamento dos Modelos
#Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

#Árvore de Decisão
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

#KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#Avaliação dos Modelos
nb_pred = nb.predict(X_test)
tree_pred = tree.predict(X_test)
knn_pred = knn.predict(X_test)

print("Precisão Naive Bayes:", accuracy_score(y_test, nb_pred))
print("Precisão Árvore de Decisão:", accuracy_score(y_test, tree_pred))
print("Precisão KNN:", accuracy_score(y_test, knn_pred))

#Graficos
precisoes = [accuracy_score(y_test, nb_pred), accuracy_score(y_test, tree_pred), accuracy_score(y_test, knn_pred)]
modelos = ['Naive Bayes', 'Árvore de Decisão', 'KNN']

plt.bar(modelos, precisoes, color=['blue', 'green', 'red'])
plt.ylabel('Precisão')
plt.title('Comparação da Precisão dos Modelos')
plt.show()

'''
#Fronteira de Decisão
# Cria uma grade de pontos em toda a região de interesse
pca = PCA(n_components=2)
tamanho_amostra = 0.1  # 10% dos dados
# Cria uma amostra aleatória do DataFrame
amostra = data.sample(frac=tamanho_amostra, random_state=42)  # Uso um valor de random_state para reproduzibilidade

# Agora você pode usar a amostra para criar seus gráficos

X_2D = pca.fit_transform(amostra)

x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))


Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

# Plote os pontos de dados originais (em 2D)
plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap=plt.cm.Paired)

plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Fronteira de Decisão após PCA')

plt.show()
'''


'''
#Naive Bayes
matriz_nb = confusion_matrix(y_test, nb_pred)
sns.heatmap(matriz_nb, annot=True, fmt='g')
plt.title('Matriz de Confusão - Naive Bayes')
plt.show()

#Arvore de Decisão
matriz_tree = confusion_matrix(y_test, tree_pred)
sns.heatmap(matriz_tree, annot=True, fmt='g')
plt.title('Matriz de Confusão - Arvore de Decisão')
plt.show()

#KNN
matriz_knn = confusion_matrix(y_test, knn_pred)
sns.heatmap(matriz_knn, annot=True, fmt='g')
plt.title('Matriz de Confusão - KNN')
plt.show()


#Curva ROC e AUC
#Naive Bayes
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_pred)
roc_auc_nb = auc(fpr_nb, tpr_nb)

plt.figure()
plt.plot(fpr_nb, tpr_nb, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Naive Bayes')
plt.legend(loc="lower right")
plt.show()

#Arvore de Decisão

fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_pred)
roc_auc_tree = auc(fpr_tree, tpr_tree)

plt.figure()
plt.plot(fpr_tree, tpr_tree, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_tree)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - Arvore de Decisão')
plt.legend(loc="lower right")
plt.show()

#KNN
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_pred)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure()
plt.plot(fpr_knn, tpr_knn, color='blue', label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC - KNN')
plt.legend(loc="lower right")
plt.show()

'''