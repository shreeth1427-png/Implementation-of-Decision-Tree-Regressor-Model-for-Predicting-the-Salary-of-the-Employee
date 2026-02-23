# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate the independent variable (X) and dependent variable (Y).
2. Create the Decision Tree Regressor model.
3. Train the model using the dataset (fit the model).
4. Predict the employee salary for new input using the trained model.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: TEJASHREE M
RegisterNumber: 212225220115
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
dataset = pd.read_csv("C:/Users/acer/Downloads/Salary (3).csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
salary_pred = regressor.predict([[6.5]])
print("Predicted Salary:", salary_pred)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
```


## Output:
<img width="779" height="565" alt="Screenshot 2026-02-23 143637" src="https://github.com/user-attachments/assets/19fa3cd4-6e1f-4dac-87e1-62eb878cd1d6" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
