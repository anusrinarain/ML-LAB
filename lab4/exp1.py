
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
url = "https://raw.githubusercontent.com/Yashappin/Machine-Learning/master/TvMarketing.csv"
sales = pd.read_csv(url)

print("First 5 rows of dataset:")
print(sales.head())

print("\nDataset Info:")
print(sales.info())

print("\nStatistical Summary:")
print(sales.describe())


plt.figure(figsize=(8,5))
plt.scatter(sales['TV'], sales['Sales'], color='blue')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.title('TV Advertising Budget vs Sales')
plt.show()


X = sales[['TV']]
y = sales['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and Testing Data Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


model = LinearRegression()
model.fit(X_train, y_train)

b0 = model.intercept_
b1 = model.coef_[0]

print("\nModel Parameters:")
print("Intercept (b0):", b0)
print("Slope (b1):", b1)

plt.figure(figsize=(8,5))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', label='Best Fit Line')
plt.xlabel('TV Advertising Budget')
plt.ylabel('Sales')
plt.title('Linear Regression Best Fit Line')
plt.legend()
plt.show()


y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    'Actual Sales': y_test.values,
    'Predicted Sales': y_pred
})

print("\nActual vs Predicted Sales:")
print(comparison.head())

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color='green')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')
plt.show()


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R^2 Score:", r2)
