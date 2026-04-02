import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_excel(r"C:\Users\Anusri\OneDrive\Documents\Desktop\T_SCADA_DEMAND_BLOCK_6Months.xlsx", sheet_name="T_SCADA_DEMAND_BLOCK")
df.drop(columns=['Solar/Non-solar'], inplace=True)

df['SCH_DATE'] = pd.to_datetime(df['SCH_DATE'], errors='coerce')
df['Hour'] = df['Hour'].astype(int)
df['DEMAND'] = pd.to_numeric(df['DEMAND'], errors='coerce')
df['Month'] = df['SCH_DATE'].dt.month
df['Day'] = df['SCH_DATE'].dt.day

latitude, longitude = 26.8467, 80.9462
start_date, end_date = "2025-01-01", "2025-06-10"

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
)

weather_df = pd.DataFrame(requests.get(url).json()['hourly'])
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df['SCH_DATE'] = weather_df['time'].dt.normalize()
weather_df['Hour'] = weather_df['time'].dt.hour + 1
weather_df.drop(columns=['time'], inplace=True)

merged_df = pd.merge(df, weather_df, on=['SCH_DATE', 'Hour'], how='inner')
merged_df.sort_values(['SCH_DATE', 'Hour'], inplace=True)
merged_df['Month'] = merged_df['SCH_DATE'].dt.month

mask = merged_df['DEMAND'].isnull()
monthly_mean = merged_df.groupby('Month')['DEMAND'].transform('mean')
merged_df['DEMAND'] = merged_df['DEMAND'].fillna(monthly_mean)

merged_df['DayOfWeek'] = merged_df['SCH_DATE'].dt.dayofweek
merged_df['IsWeekend'] = merged_df['DayOfWeek'].isin([5, 6]).astype(int)
'''merged_df['Prev_Hour_Demand'] = merged_df['DEMAND'].shift(1)
merged_df.loc[merged_df['Hour'] == 1, 'Prev_Hour_Demand'] = monthly_mean'''
merged_df.dropna(inplace=True)
clean_df = merged_df[~mask]

features = ['Hour', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'DayOfWeek', 'IsWeekend']
X, y = clean_df[features], clean_df['DEMAND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#random forest 
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred))
import numpy as np
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)

print("Random Forest R2:", r2_score(y_test, y_pred))
y_test_sorted = y_test.reset_index(drop=True)
y_pred_sorted = pd.Series(y_pred)

'''plt.figure(figsize=(10, 6))
plt.plot(y_test_sorted, label='Actual Demand', color='blue')
plt.plot(y_pred_sorted, label='Predicted Demand', color='pink')
plt.legend()
plt.title('Actual vs Predicted Demand (Line Plot)')
plt.xlabel('Sample Index')
plt.ylabel('Electricity Demand')
plt.grid(True)
plt.show()'''

'''sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Actual vs Predicted Demand")
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.tight_layout()
plt.show()'''

plt.figure(figsize=(8, 6))
sns.barplot(x=model.feature_importances_, y=features)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

monthly_avg_demand = merged_df.groupby('Month')['DEMAND'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.lineplot(x='Month', y='DEMAND', data=monthly_avg_demand, marker='o', linewidth=2)
plt.title("Average Electricity Demand per Month")
plt.xlabel("Month")
plt.ylabel("Average Demand (MW)")
import calendar
month_names = [calendar.month_name[m] for m in monthly_avg_demand['Month']]
plt.xticks(monthly_avg_demand['Month'], month_names)
plt.grid(True)
plt.tight_layout()
plt.show()
#weekday vs weekends
weekend_avg = merged_df.groupby('IsWeekend')['DEMAND'].mean().reset_index()
weekend_avg['Label'] = weekend_avg['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
plt.figure(figsize=(6, 4))
sns.barplot(x='Label', y='DEMAND', data=weekend_avg, palette='Set2')
plt.title("Average Demand: Weekday vs Weekend")
plt.ylabel("Average Demand (MW)")
plt.xlabel("")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
hourly_trend = merged_df.groupby(['Hour', 'IsWeekend'])['DEMAND'].mean().reset_index()
hourly_trend['DayType'] = hourly_trend['IsWeekend'].map({0: 'Weekday', 1: 'Weekend'})
plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly_trend, x='Hour', y='DEMAND', hue='DayType', marker='o')
plt.title("Hourly Demand: Weekday vs Weekend")
plt.grid(True)
plt.tight_layout()
plt.show()
#demand vs weather
merged_df['temp_bin'] = (merged_df['temperature_2m'] // 2) * 2
merged_df['humidity_bin'] = (merged_df['relative_humidity_2m'] // 5) * 5
merged_df['wind_bin'] = (merged_df['windspeed_10m'] // 2) * 2
temp_demand = merged_df.groupby('temp_bin')['DEMAND'].mean().reset_index()
humidity_demand = merged_df.groupby('humidity_bin')['DEMAND'].mean().reset_index()
wind_demand = merged_df.groupby('wind_bin')['DEMAND'].mean().reset_index()
plt.figure(figsize=(18, 5))
#Temperature vs Demand
plt.subplot(1, 3, 1)
sns.barplot(x='temp_bin', y='DEMAND', data=temp_demand, color='pink')
plt.title("Avg Demand vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Demand (MW)")
plt.xticks(rotation=45)
#Humidity vs Demand
plt.subplot(1, 3, 2)
sns.barplot(x='humidity_bin', y='DEMAND', data=humidity_demand, color='skyblue')
plt.title("Avg Demand vs Humidity")
plt.xlabel("Humidity (%)")
plt.ylabel("")
plt.xticks(rotation=45)
#Windspeed vs Demand
plt.subplot(1, 3, 3)
sns.barplot(x='wind_bin', y='DEMAND', data=wind_demand, color='lightgreen')
plt.title("Avg Demand vs Windspeed")
plt.xlabel("Windspeed (m/s)")
plt.ylabel("")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#correlation matrix
for month in merged_df['Month'].unique():
    plt.figure(figsize=(6, 4))
    sns.heatmap(merged_df[merged_df['Month'] == month][features + ['DEMAND']].corr(), annot=True, cmap='magma')
    plt.title(f'Correlation Matrix - {calendar.month_name[month]}')
    plt.show()
