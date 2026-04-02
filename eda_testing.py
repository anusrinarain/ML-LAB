import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df= pd.read_excel(r"C:\Users\Anusri\OneDrive\Documents\Desktop\T_SCADA_DEMAND_BLOCK_6Months.xlsx",sheet_name="T_SCADA_DEMAND_BLOCK")

sheet_name = "T_SCADA_DEMAND_BLOCK"

'''df = pd.read_excel(file_path, sheet_name=sheet_name)
df.columns = df.columns.astype(str).str.strip()'''

df['SCH_DATE'] = pd.to_datetime(df['SCH_DATE'], errors='coerce')
df['Hour'] = df['Hour'].astype(int)
df['DEMAND'] = pd.to_numeric(df['DEMAND'], errors='coerce')

df.drop(columns=['Solar/Non-solar'], inplace=True)


print("Shape of the dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:\n")
df.info()
print("\nSummary statistics:\n", df.describe())

print("\nMissing values per column:\n", df.isnull().sum())

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")


'''sns.pairplot(df[['Hour', 'DEMAND']])
plt.show()'''

corr_matrix = df.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

df['Month'] = df['SCH_DATE'].dt.month
df['Day'] = df['SCH_DATE'].dt.day

monthly_demand = df.groupby(['Month', 'Day'])['DEMAND'].mean().reset_index()
pivot_df = monthly_demand.pivot(index='Day', columns='Month', values='DEMAND')
pivot_df.columns = [calendar.month_abbr[m] for m in pivot_df.columns]

pivot_df.plot(figsize=(12,6), linewidth=2)
plt.title("Electricity Demand Trend Over Days by Month")
plt.xlabel("Day of the Month")
plt.ylabel("Average Electricity Demand")
plt.legend(title="Month")
plt.grid(True)
plt.tight_layout()
plt.show()
latitude = 26.8467
longitude = 80.9462
start_date = "2025-01-01"
end_date = "2025-06-10"

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

weather_df['Hour'] = weather_df['Hour'].astype(int)
merged_df = pd.merge(df, weather_df, on=['SCH_DATE', 'Hour'], how='inner')

merged_df['DEMAND'] = merged_df['DEMAND'].fillna(merged_df['DEMAND'].mean())
merged_df['DayOfWeek'] = merged_df['SCH_DATE'].dt.dayofweek
merged_df['IsWeekend'] = merged_df['DayOfWeek'].isin([5, 6]).astype(int)
merged_df['Prev_Hour_Demand'] = merged_df['DEMAND'].shift(1).fillna(merged_df['DEMAND'].mean())

'''sns.scatterplot(data=merged_df, x='temperature_2m', y='DEMAND')
plt.title("Demand vs Temperature")
plt.show() '''


sns.boxplot(x=df['DEMAND'])
plt.title("Outlier Detection in Demand")
plt.show()

features = ['Hour', 'temperature_2m', 'relative_humidity_2m', 'windspeed_10m', 'DayOfWeek', 'IsWeekend']

X = merged_df[features]
y = merged_df['DEMAND']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

'''sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Electricity Demand")
plt.grid(True)
plt.tight_layout()
plt.show()'''
