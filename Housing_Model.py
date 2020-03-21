# -*- coding: utf-8 -*-
"""

@author: Aayush Chaube
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report, r2_score
import statsmodels.api as sm

data=pd.read_csv("House_Data.csv")
print(data.head())

print(data.describe())

data['bedrooms'].value_counts().plot(kind='bar')
plt.title('Number of Bedroom')
plt.xlabel('Bedroom')
plt.ylabel('Count')
plt.show()
sns.despine

plt.figure(figsize=(10, 10))
sns.jointplot(x=data.lat.values, y=data.long.values, height=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()
sns.despine

plt.scatter(data.price, data.sqft_living)
plt.title("Price vs Square Feet")
plt.xlabel("Price")
plt.ylabel("Square Feet")
plt.show()
sns.despine

plt.scatter(data.price, data.long)
plt.title("Price vs Location of the area")
plt.xlabel("Price")
plt.ylabel("Longitude")
plt.show()
sns.despine

plt.scatter(data.price, data.lat)
plt.title("Latitude vs Price")
plt.xlabel("Price")
plt.ylabel("Latitude")
plt.show()
sns.despine

plt.scatter(data.bedrooms, data.price)
plt.title("Bedroom and Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter((data['sqft_living']+data['sqft_basement']), data['price'])
plt.title("Squarefeet (Overall) vs Price")
plt.xlabel("Squarefeet (Living + Basement)")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter(data.waterfront, data.price)
plt.title("Waterfront vs Price (0 = No waterfront)")
plt.xlabel("wavefront")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter(data.floors, data.price)
plt.title("Floor vs Price")
plt.xlabel("Floors")
plt.ylabel("Price")
plt.show()
sns.despine

plt.scatter(data.condition, data.price)
plt.title("Condition vs Price")
plt.ylabel("Price")
plt.xlabel("Condition")
plt.show()
sns.despine

plt.scatter(data.zipcode, data.price)
plt.title("Which is the pricey location by zipcode?")
plt.xlabel("Zipcode")
plt.ylabel("Price")
plt.show()
sns.despine

reg=LinearRegression()

labels=data['price']
conv_dates=[1 if values==2014 else 0 for values in data.date]
data['date']=conv_dates
train1=data.drop(['id', 'price'], axis=1)

x_train, x_test, y_train, y_test=train_test_split(train1, labels, test_size=0.10, random_state=2)

reg.fit(x_train, y_train)

print("Prediction (or Score): ", reg.score(x_test, y_test))

clf=ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=2, learning_rate=0.1, loss='ls')

clf.fit(x_train, y_train)

print("Boosted Prediction (or Boosted Score): ", clf.score(x_test, y_test))
