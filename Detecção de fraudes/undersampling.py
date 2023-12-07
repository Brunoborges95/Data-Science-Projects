# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:16:22 2019

@author: bruno
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_validate
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('creditcard.csv')

df = df.sample(frac=1)
fraude_df = df.loc[df['Class'] == 1]
# undersampling dos dados. FRaudes representam 10% da base
normal_df = df.loc[df['Class'] == 0][:492*9]
new_df = pd.concat([normal_df, fraude_df])

# proporção de fraudes na base
print('Normal', round(
        new_df['Class'].value_counts()[0]/len(new_df)*100, 2), '% da base')
print('Fraude', round(
        new_df['Class'].value_counts()[1]/len(new_df)*100, 2), '% da base')
# normalização robusta a outliers
RS = RobustScaler()
new_df['Amount'] = RS.fit_transform(new_df['Amount'].values.reshape(-1, 1))
new_df['Time'] = RS.fit_transform(new_df['Time'].values.reshape(-1, 1))


X1 = new_df.drop('Class', axis=1)
y1 = new_df['Class']

# gradient boosting
clf = GradientBoostingClassifier(n_estimators=100, 
                                 learning_rate=1.0, max_depth=1)

clf.fit(X1, y1)

predição_clf = clf.predict(df.drop('Class', axis=1).values)
# matriz de confusão
cm_GBC = metrics.confusion_matrix(df['Class'].values, predição_clf)


# árvores de decisão funcionam bem em dados desbalanceados
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

rfc = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, 
                               random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X1, y1)


predição_rfc = rfc.predict(df.drop('Class', axis=1).values)
cm_rfc = metrics.confusion_matrix(df['Class'].values, predição_rfc)

metrics.recall_score(df['Class'].values, predição_rfc)
metrics.precision_score(df['Class'].values, predição_rfc)
# o fbeta score=2 da mais peso ao recall (beta>1)
metrics.fbeta_score(df['Class'].values, predição_rfc, beta=3)
metrics.roc_auc_score(df['Class'].values, predição_rfc)

metrics.recall_score(df['Class'].values, predição_clf)
metrics.precision_score(df['Class'].values, predição_clf)
# o fbeta score=2 da mais peso ao recall (beta>1)
metrics.fbeta_score(df['Class'].values, predição_clf, beta=3)
metrics.roc_auc_score(df['Class'].values, predição_clf)

# determinando o score da curva ROC com cross validation
scoreclf = cross_validate(clf, X1, y1, cv=10, scoring='roc_auc', return_train_score=True)
clf_test = np.mean(scoreclf['test_score'])
clf_train = np.mean(scoreclf['train_score'])


