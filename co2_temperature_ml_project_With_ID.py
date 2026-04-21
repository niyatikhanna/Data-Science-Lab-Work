"""
Machine Learning Project
GGU ID: nkhanna898
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv("Global_CO2_Temperature_1900_2023.csv")

X = data[['CO2_Emissions_Billion_Metric_Tons']]
y = data['Global_Temperature_Anomaly_C']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

print("Linear R2:", r2_score(y_test, lr.predict(X_test)))
print("Decision Tree R2:", r2_score(y_test, dt.predict(X_test)))
