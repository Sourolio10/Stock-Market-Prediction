# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:13:05 2019

@author: Annesh Nandy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Get the stock data
data=pd.read_csv("AMZN.csv")
#Take a look
print(data.head(10))

#A variable for predicting 'n' days out into the future
forecast_out=30
#Create another column (the target) shifted 'n' units up
data['Prediction']=data[['Adj Close']].shift(-forecast_out)
#Print the new dataset
print(data.tail())

#Create the independent data set and convert the dataframe to a numpy array
X=np.array(data.iloc[:,1:5])
#Remove the last 30 rows
X=X[:-forecast_out]
print(X)

#Create the dependent data set and convert the dataframe to a numpy array
Y=np.array(data['Prediction'])
#Get all the y-values except the last 30 rows
Y=Y[:-forecast_out]
print(Y)

#Split the data into training and testing data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Create and train the support Vector Machine(Regressor)
svr_rbf=SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train,y_train)
'''
#Testing Model:Score returns the coefficient of determination R^2 of the prediction
#The best prediction score is 1.0
lr_confidence=lr.score(x_test,y_test)
print("lr confidence:",lr_confidence)
'''
#Create the train and liner Regression model
lr=LinearRegression()
#Train the model
lr.fit(x_train,y_train)

#Testing Model:Score returns the coefficient of determination R^2 of the prediction
#The best prediction score is 1.0
lr_confidence=lr.score(x_test,y_test)
print("lr confidence:",lr_confidence)

#Set x_forecast equal to the last 30 rows of the original data set from Adj Close column
x_forecast=np.array(data.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

#Print linear regression model predictions for the next 30 days
lr_prediction=lr.predict(x_forecast)
print(lr_prediction)

#Print Support Vector regressor model predictions for the next 30 days
svm_prediction=svr_rbf.predict(x_forecast)
print(svm_prediction)