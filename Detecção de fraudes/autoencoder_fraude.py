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

import warnings
import os
os.chdir('C:/Users/bruno/Documents/Machine Learning/Detecção de fraudes')
warnings.filterwarnings('ignore')

df=pd.read_csv('creditcard.csv')
df.head(5)

#descrevendo os dados
df.describe().transpose()

#visualizando se tem dados faltantes
df.isnull().sum().max()

label=['Normal', 'Fraude']

#Observe como o nosso conjunto de dados original é desequilibrado!
print('Normal', round(df['Class'].value_counts()[0]/len(df)*100,2), '% da base')
print('Normal', round(df['Class'].value_counts()[1]/len(df)*100,2), '% da base')

fig, ax=plt.subplots(1,1,figsize=(18,4))
Amount_val=df['Amount'].values
time=df['Time'].values

sns.distplot(Amount_val,color='blue')
ax.set_title('Distribuição dos quantis',fontsize=10)
ax.set_xlim([min(Amount_val),max(Amount_val)])

sns.distplot(time_value,color='green')
ax.set_title('Distribuição dos quantis',fontsize=10)
ax.set_xlim([min(time),max(time)])

fraude=df[df.Class==1]
normal=df[df.Class==0]

print('Fraudes detectadas')
fraude.Amount.describe()

print('base normal')
normal.Amount.describe()

#Gráfico de dispersão
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
f.suptitle("Hora da transação vs Valor por Classe")

ax1.scatter(fraude.Time,fraude.Amount,color='red')
ax2.scatter(normal.Time,normal.Amount,color='blue')


#vamos deixar todos os dados na mesma escala
from sklearn.preprocessing import StandardScaler, RobustScaler

robust=RobustScaler()
df['Amount']=robust.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time']=robust.fit_transform(df['Time'].values.reshape(-1,1))
df.head(4)

df=df.sample(frac=1)
fraude_df=df.loc[df['Class']==1]
normal_df=df.loc[df['Class']==0][:492]
new_df=pd.concat([normal_df,fraude_df])

from string import ascii_letters
corr=new_df.corr()
mask=np.zeros_like(corr,dtype=np.bool)
mask[np.triu_indices_from(mask)]=True
cmap=sns.diverging_palette(220,10,as_cmap=True)

#Mapa de calor para identificar correlação
fig, ax=plt.subplots(figsize=(22,15))
sns.heatmap(corr,annot=True,linewidths=.5,ax=ax,fmt='.1f',cmap=cmap,mask=mask)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

X=df.drop('Class',axis=1)
y=df['Class']

#Estratificação dos dados
SSS=StratifiedShuffleSplit(n_splits=5,test_size=.2,random_state=42)

for train_index, test_index in SSS.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test=y.iloc[test_index], y.iloc[test_index]
    
X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values

#Aplicaremos o Autoencoder
fraude_df=df.loc[df['Class']==1]
normal_df=df.loc[df.Class==0][:100000]
merge_df=pd.concat([fraude_df,normal_df])

X_train,X_test=train_test_split(merge_df,test_size=.2, random_state=12)
#A base de treinamento são todas as instâncias rotuladas com 0 (sem o rótulo)
X_train = X_train[X_train.Class==0]
X_train=X_train.drop(['Class'],axis=1)
X_train=X_train.values

#A base de teste e a classes
y_test=X_test['Class']
X_test=X_test.drop(['Class'],axis=1)
X_test=X_test.values

from sklearn.ensemble import IsolationForest
np.random.seed(42)

model = IsolationForest(random_state=42, n_jobs=4, max_samples=X_train.shape[0], bootstrap=True, n_estimators=50)
model.fit(X_train)
print(model.decision_function(valid[valid['Class'] == 0].drop('Class', axis=1).values).mean())
print(model.decision_function(valid[valid['Class'] == 1].drop('Class', axis=1).values).mean())

tresholds = np.linspace(-.2, .2, 200)
y_scores = model.decision_function(X_test)
scores = []
for treshold in tresholds:
    y_hat = (y_scores < treshold).astype(int)
    scores.append([metrics.recall_score(y_pred=y_hat, y_true=y_test),
                 metrics.precision_score(y_pred=y_hat, y_true=y_test),
                 metrics.fbeta_score(y_pred=y_hat, y_true=y_test, beta=2)])

scores = np.array(scores)
print(scores[:, 2].max(), scores[:, 2].argmax())
plt.plot(tresholds, scores[:, 0], label='$Recall$')
plt.plot(tresholds, scores[:, 1], label='$Precision$')
plt.plot(tresholds, scores[:, 2], label='$F_2$')
plt.ylabel('Score')
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.show()
opt_threshold=scores[:,2].argmax()
opt_threshold1=tresholds[opt_threshold]

input_dim = X_train.shape[1]
encoding_dim = 10

Input_layer=Input(shape=(input_dim,))
#encoder
encoder=Dense(units=encoding_dim,activation='relu', activity_regularizer=regularizers.l1(10e-6))(Input_layer)
encoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)
encoder=Dense(units=int(encoding_dim/4), activation='relu')(encoder)


#decoder
encoder=Dense(units=int(encoding_dim/4), activation='relu')(encoder)
decoder=Dense(units=int(encoding_dim/2), activation='relu')(encoder)
decoder=Dense(units=input_dim,activation='relu')(decoder)

#modelo
autoencoder=Model(inputs=Input_layer,outputs=decoder)

epochs=200
batch_size=128
model=ModelCheckpoint(X_train, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#TensorBoard=TensorBoard(log_dir='./logs',histogram_freq=0,write_grads=True,write_images=True)

autoencoder.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
2
predictions=autoencoder.predict(X)

scores = np.array(scores)
print(scores[:, 2].max(), scores[:, 2].argmax())
plt.plot(tresholds, scores[:, 0], label='$Recall$')
plt.plot(tresholds, scores[:, 1], label='$Precision$')
plt.plot(tresholds, scores[:, 2], label='$F_2$')
plt.ylabel('Score')
plt.xlabel('Threshold')
plt.legend(loc='best')
plt.show()
opt_threshold=scores[:,2].argmax()
opt_threshold1=tresholds[opt_threshold]



plt.figure()
plt.plot(history['loss'])
plt.plot(history['val_loss'])
y_test=np.array(y_test)

mse = np.mean(np.power(y.values.reshape(-1,1) - predictions, 2), axis=1)

mse=np.array(mse)
error_df=pd.DataFrame({'reconstruction_error':mse,'true_class':y})
error_df.dtypes
import sklearn.metrics as metrics
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

import sklearn.metrics as metrics
fpr, tpr, limiar = metrics.roc_curve(y_test, mse)
plt.plot(fpr,tpr,label='ROC')


table_auc=pd.DataFrame({'fpr':fpr,'tpr':tpr,'limiar':limiar})


#limiar=3.11
for i in limiar:
    error_df.loc[error_df.reconstruction_error>=i,'reconstruction_error2']=1
    error_df.loc[error_df.reconstruction_error<i,'reconstruction_error2']=0
    scores.append(metrics.fbeta_score(y_true=error_df['true_class'],y_pred=error_df['reconstruction_error2'],beta=2))
    scores.append(metrics.confusion_matrix(y_true=error_df['true_class'],y_pred=error_df['reconstruction_error2']))
opt=np.array(scores).argmax()
opt_tre=tresholds[opt]

error_df.loc[error_df.reconstruction_error>=opt_tre,'reconstruction_error2']=1
error_df.loc[error_df.reconstruction_error<opt_tre,'reconstruction_error2']=0

metrics.confusion_matrix(error_df['true_class'],error_df['reconstruction_error2'])
a=pd.value_counts(table_auc2['true_class'])
b=pd.value_counts(error_df['true_class'])






