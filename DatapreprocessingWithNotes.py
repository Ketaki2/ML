# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:41:10 2018

@author: Ketaki Vaidya
"""

#indexes in python start at 0
#ML models are mathematical models
import numpy as np #for mathematics
import matplotlib.pyplot as plt #charts plotting
import pandas as pd#for importing and managing datasets

dataset=pd.read_csv('Data.csv')  #after setting the working direcctory
X=dataset.iloc[:,:-1].values #iloc[rows,columns] to fetch those many rows and columns frm the dataset
y=dataset.iloc[:,3] #X=Independent vairable,y=dependent variable


#Replacing missing data with mean of the values in that column
from sklearn.preprocessing import Imputer #Imputer is a class from the library
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0) #Parenthesis around class because v r calling it
#imputer=object of Imputer class
#press ctrl+I over imputer to know abt it
imputer.fit(X[:,1:3]); #We take columns 1 and 2.1:3 is written because upper bound is excluded
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder #Encodes values without bothering abt the order
labelEncoder_X=LabelEncoder()
X[:,0]=labelEncoder_X.fit_transform(X[:,0]) #this has assigned 0 to france,1 to Germany and 2 to spain
#But this may suggest that Spain has a higher priority over Germany which is nt correct
#So we use Dummy encoding
onehotencoder = OneHotEncoder(categorical_features=[0]) #3 columns habing 1 for the country which that row belongs to
X=onehotencoder.fit_transform(X).toarray();
labelEncoder_y=LabelEncoder()
y=labelEncoder_y.fit_transform(y);

#we should split the dataset into training set and test set.Performance shud b same over both so that we understand
#if the correlations between the dataset(to make predictions later) have been correctly understood by the model
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
#generally test_size shud b 20%,25% 

#We are establishing correlation between X(matrix of features) and y(dependent variable)-purchased or nt

#most ML models are based on euclidean distance between the dependent and independent variable
#So feature scaling is necessary
#We can do standardisation and normalization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#When applying StandardScaler set to training set,you first need to fit and then transform
#for training set,only transform because its already fitted to training set
X_test = sc_X.transform(X_test)
#we dnt need to scale dummy variables else u might lose interpretation...but generally depends on the context.
#we are scaling them here because no interpretation is to be done ater this

#decision trees are nt based on euclidean distance
#We dnt need to apply scaling to dependent variable in this case as it is categorical in this case and this is a classification problem
#but for regression when dependent variable will take huge range of values,we will hv to scale it as well







