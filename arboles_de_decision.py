import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,plot_confusion_matrix
from sklearn import tree

df=pd.read_csv('DataSet_Titanic.csv')
print(df)

df.head()
print(df.head())
X=df.drop('Sobreviviente',axis=1) 
#drop para eliminar 
#axis=1 es una columna 
#axis=0 es una fila 
y=df.Sobreviviente

print(X.head())
print(y.head())
arbol=DecisionTreeClassifier(max_depth=2,random_state=42)
arbol.fit(X,y)  #entrenamos la maquina 
pred_y=arbol.predict(X)
print('Precision:', accuracy_score(pred_y,y))
#matrix de confusion 
print(confusion_matrix(y, pred_y))
#---------------------------------------------
plot_confusion_matrix(arbol, X ,y, cmap=plt.cm.Blues,values_format='.0f')
#-------------------------------------------
plot_confusion_matrix(arbol, X ,y, cmap=plt.cm.Blues,values_format='.2f', normalize='true')
#-------------------------------------------
plt.figure(figsize=(10,8))
tree.plot_tree(arbol, filled=True, feature_names=X.columns)
plt.show()
#---------------------------------------------
importancias=arbol.feature_importances_
columnas=X.columns
sns.barplot(columnas, importancias)
plt.title('importancia de cada atributo')
plt.show()