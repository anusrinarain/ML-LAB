
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("vehicle_co2_dataset.csv")

print("\nDataset Shape:", df.shape)
print("\nColumn Names:\n", df.columns)

df = df[['engine_cc', 'vehicle_weight', 'co2_g_km']]
df.columns = ['Volume', 'Weight', 'CO2']

print("\nSelected Data:\n", df.head())
corr = df.corr()
print("\nCorrelation Matrix:\n", corr)

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.boxplot(y=df['Volume'])
plt.title("Outliers in Volume")

plt.subplot(1, 3, 2)
sns.boxplot(y=df['Weight'])
plt.title("Outliers in Weight")

plt.subplot(1, 3, 3)
sns.boxplot(y=df['CO2'])
plt.title("Outliers in CO2")

plt.tight_layout()
plt.show()

plt.scatter(df['Volume'], df['CO2'])
plt.xlabel("Volume (Engine CC)")
plt.ylabel("CO2 (g/km)")
plt.title("CO2 vs Volume")
plt.show()

plt.scatter(df['Weight'], df['CO2'])
plt.xlabel("Weight")
plt.ylabel("CO2 (g/km)")
plt.title("CO2 vs Weight")
plt.show()

X = df[['Volume', 'Weight']]
y = df['CO2']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

print("\nRegression Coefficients:")
print("Volume Coefficient:", model.coef_[0])
print("Weight Coefficient:", model.coef_[1])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

plt.plot(y_test.values, label="Actual CO2")
plt.plot(y_pred, label="Predicted CO2")
plt.title("Actual vs Predicted CO2 Emission")
plt.xlabel("Test Samples")
plt.ylabel("CO2")
plt.legend()
plt.show()
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
