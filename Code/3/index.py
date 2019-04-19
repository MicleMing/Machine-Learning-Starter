from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../../datasets/50_Startups.csv')

X = dataset.iloc[:, : -1].values
Y = dataset.iloc[:, 4].values


labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)


regressor = LinearRegression()
regressor.fit(X_test, Y_test)
y_pred = regressor.predict(X_test)

print(len(X_train))
print(len(Y_train))
