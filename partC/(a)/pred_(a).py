import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import joblib
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split


train = pd.read_csv("train_acc.csv")

test = pd.read_csv("test_acc.csv")

'''
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(train.iloc[2:])
train = imputer.transform(train.iloc[2:])
'''
train_x = train.iloc[:,2:-1].values
train_y = train.iloc[:,-1].values

print(train_x)

test_x = test.iloc[:,2:-1].values
test_y = test.iloc[:,-1].values

le = LabelEncoder() #實例化
print(train_y)
train_y = le.fit_transform(train_y)
test_y = le.fit_transform(test_y)
'''
ohe = OneHotEncoder()
train_y = ohe.fit_transform(train_y)
test_y = ohe.fit_transform(test_y).toarray()
'''


from xgboost import XGBClassifier
xgbc = XGBClassifier()


xgbc.fit(train_x, train_y)
print('training score: ' + str(xgbc.score(train_x, train_y)) )

predict = xgbc.predict(test_x)
print('test score: ' + str(xgbc.score(test_x, test_y)))
test['encoder'] = test_y
test['pred'] = predict
test.to_csv('pred_ouptut.csv', index = False)
