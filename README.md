# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset containing house features such as area, number of bedrooms, and age of the house along with output values (price and occupants).
2. Split the dataset into training and testing sets and standardize the input features using StandardScaler.
3. Train the SGD Regressor model using MultiOutputRegressor for predicting multiple output variables.
4. Predict the house price and number of occupants for test data and evaluate the model using Mean Squared Error and R² score.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: ALAGESHWARI V
RegisterNumber:  212224240010
*/
```

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 4200, 5000],
    'Bedrooms': [3, 4, 3, 5, 4, 6, 5, 7],
    'Age': [10, 15, 20, 8, 12, 5, 7, 3],
    'Price': [300000, 400000, 500000, 600000, 650000, 700000, 720000, 800000],
    'Occupants': [4, 5, 4, 6, 5, 7, 6, 8]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df[['Price', 'Occupants']]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted [Price, Occupants]:")
print(y_pred)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)
```

## Output:

<img width="490" height="172" alt="image" src="https://github.com/user-attachments/assets/f101f253-0ce8-473b-8760-410237adf5e9" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
