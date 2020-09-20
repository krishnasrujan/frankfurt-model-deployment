# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:34:52 2020

@author: Srujan
"""
import pandas as pd
import quandl,math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split
from matplotlib import style
from sklearn.linear_model import Ridge
style.use('ggplot')
import pickle

#Frankfurt stock exchange
df= pd.read_csv('C:/Users/Srujan/Documents/Datasets/frankfurt_stock_exchange.csv',parse_dates=['Date'],index_col='Date')
df.drop(columns=['Change','Last Price of the Day','Daily Traded Units','Daily Turnover'],inplace=True)


for feat in df.columns:
    df[feat].fillna(df[feat].mean())
    
df['HL_PCT']=(df['High']-df['Close'])/df['Close']*100
df['PCT_change']=(df['Close']-df['Open'])/df['Open']*100

df=df[['Close','HL_PCT','PCT_change']]
forecast_col='Close'
forecast_out=5
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


train_count =int(df.shape[0]*0.85)
test_count = df.shape[0]-train_count

y_train = df.iloc[:train_count,3]
y_test  = df.iloc[train_count:,3]

x_train = df.iloc[:train_count,0:-1]
x_test = df.iloc[train_count:,0:-1]

x_lately=x_train[-forecast_out:]
x_train=x_train[:-forecast_out]
df.dropna(inplace=True)
y_train=y_train[:-forecast_out]

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)



clf=Ridge()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
forecast_set=clf.predict(x_lately)
print(forecast_set,forecast_out)

pickle.dump(clf,open('frankfurt_model.pkl','wb'))