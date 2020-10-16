# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:04:52 2020

@author: RADHIKA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#reading data from remote link
url="http://bit.ly/w-data"
d1=pd.read_csv(url)
print("data imported successfully")
d1.head(10)
d1.plot(x="Hours",y="Scores",style='^',color='red')
plt.title("Hours VS Percentage")
plt.xlabel("hours studied")
plt.ylabel("percentage Score")
plt.show()

X=d1.iloc[:,:-1].values
y=d1.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

line=regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line,color='red')
plt.show()
print(X_test)
y_pred=regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df
regressor.predict([[8.25]])
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
