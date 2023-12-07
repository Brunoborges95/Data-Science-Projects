# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:53:03 2019

@author: bruno
"""

import numpy as np
import pandas as pd

df = pd.DataFrame([10, 20, 30, 40], columns=['numbers'],
index=['a', 'b', 'c', 'd'])

df.index
df.loc['c']
df.loc[df.index[1:3]]
df.sum()

df.apply(lambda x: x ** 2)
df**2

df['floats'] = (1.5, 2.5, 3.5, 4.5)
# new column is generated
df['names'] = pd.DataFrame(['Yves', 'Guido', 'Felix', 'Francesc'],
index=['d', 'a', 'b', 'c'])

df = df.append(pd.DataFrame({'numbers': 100, 'floats': 5.75,
'names': 'Henry'}, index=['z',]))
df = df.join(pd.DataFrame([1, 4, 9, 16, 25],
index=['a', 'b', 'c', 'd', 'y'],
columns=['squares',]),
how='outer')

df[['numbers', 'squares']].mean()

a = np.random.standard_normal((9, 4))
a.round(6)

df = pd.DataFrame(a)
dates = pd.date_range('2015-1-1', periods=9, freq='M')
dates
df.columns=[['n1','n2','n3','n4']]
df.index=dates

df.describe()

import matplotlib.pyplot as plt
df['n1'].cumsum().plot(style='r', lw=2.)
plt.xlabel('date')
plt.ylabel('value')

S0 = 100 # initial value
r = 0.05 # constant short rate
sigma = 0.25 # constant volatility
T = 2.0 # in years
I = 10000 # number of random draws
ST1 = S0 * np.exp((r - 0.5 * sigma ** 2) * T
+ sigma * np.sqrt(T) * npr.standard_normal(I))