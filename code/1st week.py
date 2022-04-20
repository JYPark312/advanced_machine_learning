# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 13:11:07 2021

@author: young
"""

import pandas as pd\
from sklearn.linear_model import LinearRegression    

petrol = pd.read_csv("https://drive.google.com/uc?export=download&id=1R9B0D_fSjfCiSaS1WWEjEbSHFOXlVjQA")

X = petrol[['tax', 'income', 'highway', 'license']]
y=petrol['consumption']

lr=LinearRegression()

lr.fit(X,y)

lr.coef_ #coefficient
lr.intercept_ #beta

y_pred = lr.predict(X) #예측값

error = y-y_pred #error값

lr.score(X, y)


