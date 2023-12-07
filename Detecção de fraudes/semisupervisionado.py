# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:41:01 2019

@author: bruno
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pylab import rcParams
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input,Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')

X = df.drop('Class',axis=1)
y = df['Class']


fraude_df = df.loc[df['Class'] == 1]
normal_df = df.loc[df.Class == 0]
merge_df = pd.concat([fraude_df, normal_df])


X_train, y, _, _ = train_test_split(normal_df, normal_df, test_size=.2, random_state=42)

y_valid, y_test, _, _ = train_test_split(y, y, test_size=.5, random_state=42)
y_anormal_val, y_anormal_test, _, _ = train_test_split(fraude_df, fraude_df, test_size=.5, random_state=42)

#a base treino contem apenas dados normais, enquanto a base de validação e teste contem dados com fraudes
train = X_train.reset_index(drop=True) #aqui o modelo se ajusta aos dados
valid = y_valid.append(y_anormal_val).sample(frac=1).reset_index(drop=True) #base de vakidação
test = y_test.append(y_anormal_test).sample(frac=1).reset_index(drop=True) #base de teste

print('Train shape: ', train.shape)
print('Proportion os anomaly in training set: %.2f\n' % train['Class'].mean())
print('Valid shape: ', valid.shape)
print('Proportion os anomaly in validation set: %.2f\n' % valid['Class'].mean())
print('Test shape:, ', test.shape)
print('Proportion os anomaly in test set: %.2f\n' % test['Class'].mean())

# ajuste da curva gausiana
from scipy.stats import multivariate_normal
mu = train.drop('Class', axis=1).mean(axis=0).values
sigma = train.drop('Class', axis=1).cov().values
model_mn = multivariate_normal(cov=sigma, mean=mu, allow_singular=True)
prediction=model_mn.logpdf(valid.drop('Class',axis=1).values)

# definindo o limiar
tresholds = np.linspace(-1000,-10, 150)
scores = []
for treshold in tresholds:
    y_hat = (model_mn.logpdf(valid.drop('Class', axis=1).values) < treshold).astype(int)
    scores.append([metrics.recall_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.precision_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.fbeta_score(y_pred=y_hat, y_true=valid['Class'].values, beta=2)])

scores = np.array(scores)
print(scores[:, 2].max(), scores[:, 2].argmax())
plt.plot(tresholds, scores[:, 0], label='$Recall$')
plt.plot(tresholds, scores[:, 1], label='$Precision$')
plt.plot(tresholds, scores[:, 2], label='$F_2$')
plt.ylabel('Score')

# plt.xticks(np.logspace(-10, -200, 3))
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.show()

final_tresh = tresholds[scores[:, 2].argmax()]
y_hat_test = (model_mn.logpdf(X.values) < final_tresh).astype(int)

print('Final threshold: %.4f' % final_tresh)
print('Test Recall Score: %.3f' % metrics.recall_score(y_pred=y_hat_test, y_true=y.values))
print('Test Precision Score: %.3f' % metrics.precision_score(y_pred=y_hat_test, y_true=y.values))
print('Test F2 Score: %.3f' % metrics.fbeta_score(y_pred=y_hat_test, y_true=y.values, beta=2))
cnf_matrix_mn = metrics.confusion_matrix(y.values, y_hat_test)


X_train = X_train[X_train.Class==0]
X_train=X_train.drop(['Class'],axis=1)
X_train=X_train.values

X_test=X_test.drop(['Class'],axis=1)
X_test=X_test.values

y_test=X_test['Class']


from sklearn.ensemble import IsolationForest
np.random.seed(42)

model_IF = IsolationForest(random_state=42, n_jobs=4, max_samples=train.shape[0], bootstrap=True, n_estimators=50)
model_IF.fit(train.drop('Class', axis=1).values)
tresholds = np.linspace(-.2, .2, 200)
prediction_IF = model_IF.decision_function(valid.drop('Class', axis=1).values)
scores = []
confusionmatrix=[]
for treshold in tresholds:
    y_hat = (prediction_IF < treshold).astype(int)
    scores.append([metrics.recall_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.precision_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.fbeta_score(y_pred=y_hat, y_true=valid['Class'].values, beta=2)])
    confusionmatrix.append(metrics.confusion_matrix(y_pred=y_hat, y_true=valid['Class'].values))
confusionmatrix[0][0,0]
scores=np.array(scores)
a=scores[:,2].argmax()\\
opt_trh=tresholds[a]
y_hat = (prediction_IF < opt_trh).astype(int)
cm_IF=metrics.confusion_matrix(valid['Class'].values,y_hat)
metrics.roc_auc_score(y_test,y_hat)


plt.plot(tresholds, scores[:, 0], label='$Recall$')
plt.plot(tresholds, scores[:, 1], label='$Precision$')
plt.plot(tresholds, scores[:, 2], label='$F_2$')
plt.ylabel('Score')
# plt.xticks(np.logspace(-10, -200, 3))
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.show()

final_tresh = tresholds[scores[:, 2].argmax()]
y_hat_test = (model_IF.decision_function(X.values) < final_tresh).astype(int)

print('Final threshold: %.4f' % final_tresh)
print('Test Recall Score: %.3f' % metrics.recall_score(y_pred=y_hat_test, y_true=y.values))
print('Test Precision Score: %.3f' % metrics.precision_score(y_pred=y_hat_test, y_true=y.values))
print('Test F2 Score: %.3f' % metrics.fbeta_score(y_pred=y_hat_test, y_true=y.values, beta=2))



cnf_matrix = metrics.confusion_matrix(y.values, y_hat_test)

Lucro_IF=19915*20-33*100-78*20
Lucro_GBC=10448*20-2*100-9545*20

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, n_init=4, random_state=42)
gmm.fit(train.drop('Class', axis=1).values)
tresholds = np.linspace(-400, 0, 100)
y_scores = gmm.score_samples(valid.drop('Class', axis=1).values)
scores = []
for treshold in tresholds:
    y_hat = (y_scores < treshold).astype(int)
    scores.append([metrics.recall_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.precision_score(y_pred=y_hat, y_true=valid['Class'].values),
                 metrics.fbeta_score(y_pred=y_hat, y_true=valid['Class'].values, beta=2)])

scores = np.array(scores)
print(scores[:, 2].max(), scores[:, 2].argmax())

final_tresh = tresholds[scores[:, 2].argmax()]
y_hat_test = (gmm.score_samples(X.values) < final_tresh).astype(int)

print('Final threshold: %.4f' % final_tresh)
print('Test Recall Score: %.3f' % metrics.recall_score(y_pred=y_hat_test, y_true=y.values))
print('Test Precision Score: %.3f' % metrics.precision_score(y_pred=y_hat_test, y_true=y.values))
print('Test F2 Score: %.3f' % metrics.fbeta_score(y_pred=y_hat_test, y_true=y.values, beta=2))

cnf_gmm = metrics.confusion_matrix(y.values, y_hat_test)

input_dim = train.drop('Class', axis=1).values.shape[1]
encoding_dim = 10

Input_layer=Input(shape=(input_dim,))
#encoder
encoder=Dense(units=encoding_dim,activation='relu', activity_regularizer=regularizers.l1(10e-6))(Input_layer)
encoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)


#decoder
decoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)
decoder=Dense(units=input_dim,activation='relu')(decoder)

#modelo
autoencoder=Model(inputs=Input_layer,outputs=decoder)

epochs=100
batch_size=32
model=ModelCheckpoint(train.drop('Class', axis=1).values, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#TensorBoard=TensorBoard(log_dir='./logs',histogram_freq=0,write_grads=True,write_images=True)

autoencoder.compile(optimizer='Adam',loss='mean_squared_error',metrics=['accuracy'])
history=autoencoder.fit(train.drop('Class', axis=1).values,train.drop('Class', axis=1).values,epochs=epochs,batch_size=batch_size,shuffle=True,validation_data=(y.drop('Class', axis=1).values,y.drop('Class', axis=1).values),verbose=1).history

predictions=autoencoder.predict(test.drop('Class', axis=1).values)

y_scores.min()

tresholds = np.linspace(-20, 2000, 200)
scores = []
mse = np.mean(np.power(test.drop('Class', axis=1).values - predictions, 2), axis=1)

for treshold in tresholds:
    y_hat = (mse < treshold).astype(int)
    scores.append([metrics.recall_score(y_pred=y_hat, y_true=test['Class'].values),
                 metrics.precision_score(y_pred=y_hat, y_true=test['Class'].values),
                 metrics.fbeta_score(y_pred=y_hat, y_true=test['Class'].values, beta=2)])
scores = np.array(scores)
print(scores[:, 2].max(), scores[:, 2].argmax

final_tresh = tresholds[scores[:, 2].argmax()]

y_hat_test = (autoencoder.predict(X.values).mean(axis=1) < final_tresh).astype(int)
print('Final threshold: %.4f' % final_tresh)
print('Test Recall Score: %.3f' % metrics.recall_score(y_pred=y_hat_test, y_true=y.values))
print('Test Precision Score: %.3f' % metrics.precision_score(y_pred=y_hat_test, y_true=y.values))
print('Test F2 Score: %.3f' % metrics.fbeta_score(y_pred=y_hat_test, y_true=y.values, beta=2))

cnf_ae = metrics.confusion_matrix(y.values, y_hat_test)

predictions=autoencoder.predict(X_test)


plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
y_test=np.array(y_test)
mse=np.array(mse)
error_df=pd.DataFrame({'reconstruction_error':mse,'true_class':y_test})
error_df.dtypes
import sklearn.metrics as metrics
# Compute ROC curve and ROC area for each class

fpr, tpr, limiar = metrics.roc_curve(y_test, mse)
plt.plot(fpr,tpr,label='ROC')


table_auc=pd.DataFrame({'fpr':fpr,'tpr':tpr,'limiar':limiar})
#limiar=3.11

error_df.loc[error_df.reconstruction_error>=3.6,'reconstruction_error2']=1
error_df.loc[error_df.reconstruction_error<3.6,'reconstruction_error2']=0
auc=metrics.roc_auc_score(error_df['true_class'],error_df['reconstruction_error'])

metrics.confusion_matrix(error_df['true_class'],error_df['reconstruction_error2'])