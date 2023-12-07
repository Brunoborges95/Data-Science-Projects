# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 13:54:38 2018

@author: bruno
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

base=pd.read_csv('agrupamento.csv').values
iniciais=pd.read_csv('centroides_iniciais.csv').values
#Usando o KMeans 
wcss = [] #verificar o 'cotovelo'

for i in range(1, 16):
    kmeans = KMeans(n_clusters = i, random_state = 0,verbose=1,init=iniciais[0:i,:])
    kmeans.fit(base)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 16), wcss)
plt.xlabel('Número de clusters')
plt.ylabel('WCSS')
#visualização gráfica
plt.scatter(base[:,1],base[:,2])

silhouette=[] #verificar a silhoueta
for i in range(2, 16):
    kmeans = KMeans(n_clusters = i, random_state = 0,verbose=1,init=iniciais[0:i,:])
    kmeans.fit(base)
    labels=kmeans.labels_
    silhouette.append(metrics.silhouette_score(base,labels,metric='euclidean'))
plt.plot(range(2, 16), silhouette)
plt.xlabel('Número de clusters')
plt.ylabel('silhouette')
#visualização gráfica
plt.scatter(base[:,1],base[:,2])

CHI=[]
for i in range(2, 16):
    kmeans = KMeans(n_clusters = i, random_state = 0,verbose=1,init=iniciais[0:i,:])
    kmeans.fit(base)
    labels=kmeans.labels_
    CHI.append(metrics.calinski_harabaz_score(base,labels))
plt.plot(range(2, 16), CHI)
plt.xlabel('Número de clusters')
plt.ylabel('CHI')

#o número ideal de clusters é 8

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
import random
import pandas as pd
import numpy as np

random.seed ( 42 )
base1=pd.read_csv('classificacao_1.csv')
base2=pd.read_csv('classificacao_2.csv')
base3=pd.read_csv('regressao_1.csv')
base4=pd.read_csv('regressao_2.csv')

X1 = base1.iloc[:, 0:18].values
Y1=base1.iloc[:,18].values
X2=base2.iloc[:, 0:12].values
Y2=base2.iloc[:, 12].values
X3=base3.iloc[:, 0:16].values
Y3=base3.iloc[:, 16].values
X4=base4.iloc[:, 0:10].values
Y4=base4.iloc[:,10].values

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier #For Classification
from sklearn.ensemble import GradientBoostingRegressor
#Algoritmos configurados conforme teste
classificadorNB = GaussianNB() #Naive Bayes

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)

neigh = KNeighborsRegressor(n_neighbors=100) #KNN
EN=ElasticNet(alpha=.01, copy_X=True, fit_intercept=True, l1_ratio=0.5,
      max_iter=1000, normalize=False, positive=False, precompute=False,
      random_state=0, selection='cyclic', tol=0.0001, warm_start=False) #Elastic Net
classificadorSVM = SVC(kernel = 'linear', C = 1,verbose=1) #SVM
from sklearn import metrics
scoreSVM = cross_validate(classificadorSVM, X1, Y1, cv=10, scoring='f1',return_train_score=True)
scoreSVM_test=np.mean(scoreSVM['test_score']) #score de teste SVM
scoreSVM_train=np.mean(scoreSVM['train_score']) #score de treino SVM

scoreNB = cross_validate(classificadorNB, X2, Y2, cv=5, scoring='roc_auc',return_train_score=True)
scoreNB_test=np.mean(scoreNB['test_score']) #score de treino Naive Bayes
scoreNB_train=np.mean(scoreNB['train_score']) #score de teste Naive Bayes

scoreclf = cross_validate(clf, X2, Y2, cv=5, scoring='roc_auc',return_train_score=True)
scoreclf_test=np.mean(scoreNB['test_score']) #score de treino Naive Bayes
scoreclf_train=np.mean(scoreNB['train_score']) #score de teste Naive Bayes

scoreKNN = cross_validate(neigh, X3, Y3, cv=5, scoring='r2',return_train_score=True)
scoreKNN_test=np.mean(scoreKNN['test_score']) #score de treino KNN
scoreKNN_train=np.mean(scoreKNN['train_score']) #score de teste KNN

scoreEN = cross_validate(EN, X4, Y4, cv=10, scoring='neg_mean_squared_error',return_train_score=True)
scoreEN_test=np.mean(scoreEN['test_score']) #score de treino Elastic Net
scoreEN_train=np.mean(scoreEN['train_score']) #score de teste Elastic NEt
#Para Elastic Net, o MSE da validação é 102.39886
#Para Elastic Net, o MSE de treino é 100.2032
#Para SVM,o score de validação é 0.8122
#Para o SVM,o score de treino é 0.8224
#Para o NB, o score de validação 0.8557
#Para o NB, o score de treino é 0.86
#Para KNN, o score de teste é 0.6346
#Para o KNN,o score de trino é 0.6427
classificadorSVM.fit(X, Y)
base=pd.read_csv('classificacao_1.csv')



previsores=base
a=sorted(metrics.SCORERS.keys())