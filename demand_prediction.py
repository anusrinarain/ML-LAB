import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

df = pd.read_excel(r"C:\Users\Anusri\OneDrive\Documents\Desktop\T_SCADA_DEMAND_BLOCK_6Months.xlsx",
                   sheet_name="T_SCADA_DEMAND_BLOCK")
if 'Solar/Non-solar' in df.columns:
    df.drop(columns=['Solar/Non-solar'], inplace=True, errors='ignore')
df['SCH_DATE'] = pd.to_datetime(df['SCH_DATE'], errors='coerce')
df['Hour'] = df['Hour'].astype(int)
df['DEMAND'] = pd.to_numeric(df['DEMAND'], errors='coerce')
latitude, longitude = 26.8467, 80.9462
start_date, end_date = "2025-01-01", "2025-06-10"
url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
)
response = requests.get(url)
data = response.json()
weather_df = pd.DataFrame(data['hourly'])

weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df['SCH_DATE'] = weather_df['time'].dt.normalize()
weather_df['Hour'] = weather_df['time'].dt.hour + 1
weather_df.drop(columns=['time'], inplace=True)
merged_df = pd.merge(df, weather_df, on=['SCH_DATE', 'Hour'], how='inner')
merged_df.sort_values(['SCH_DATE', 'Hour'], inplace=True)
merged_df['Month'] = merged_df['SCH_DATE'].dt.month
merged_df['DEMAND'] = merged_df.groupby('Month')['DEMAND'].transform(lambda x: x.fillna(x.mean()))
merged_df['DayOfWeek'] = merged_df['SCH_DATE'].dt.dayofweek
merged_df['IsWeekend'] = merged_df['DayOfWeek'].isin([5, 6]).astype(int)
merged_df['Hour_sin'] = np.sin(2 * np.pi * merged_df['Hour'] / 24)
merged_df['Hour_cos'] = np.cos(2 * np.pi * merged_df['Hour'] / 24)


features = ['Hour_sin', 'Hour_cos', 'temperature_2m', 'relative_humidity_2m',
            'windspeed_10m', 'DayOfWeek', 'IsWeekend']
X = merged_df[features]
y = merged_df['DEMAND']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Improved Linear Regression Results:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", np.sqrt(mse))
print("R² Score:", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Linear Regression: Actual vs Predicted Demand")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(y_test.values[:200], label='Actual', color='blue')
plt.plot(y_pred[:200], label='Predicted', color='orange')
plt.xlabel("Sample Index")
plt.ylabel("Demand")
plt.title("Linear Regression: Actual vs Predicted (Line Plot)")
plt.legend()
plt.grid(True)
plt.show()
