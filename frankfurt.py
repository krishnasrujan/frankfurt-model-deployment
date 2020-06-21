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
from sklearn.ensemble import RandomForestRegressor
style.use('ggplot')
import pickle

#Frankfurt stock exchange
df=quandl.get("FSE/ZO1_X", authtoken="8PALk3sJpg7KuJtoyuSV")
df.drop(columns=['Change','Last Price of the Day','Daily Traded Units','Daily Turnover'],inplace=True)

df.fillna(-99999,inplace=True)
#for feat in df.columns:
   # df[feat].fillna(df[feat].median)
df['HL_PCT']=(df['High']-df['Close'])/df['Close']*100
df['PCT_change']=(df['Close']-df['Open'])/df['Open']*100

df=df[['Close','HL_PCT','PCT_change']]
forecast_col='Close'
forecast_out=int(math.ceil(0.01*len(df)))
print(forecast_out)
df['label']=df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X=np.array(df.drop(['label'],1))
X_lately=X[-forecast_out:]
X=X[:-forecast_out]
df.dropna(inplace=True)
y=np.array(df['label'])
y=y[:-forecast_out]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


clf=RandomForestRegressor()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)
forecast_set=clf.predict(X_lately)
print(forecast_set,forecast_out)

pickle.dump(clf,open('frankfurt_model.pkl','wb'))