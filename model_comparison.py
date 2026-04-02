import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
import calendar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
})

FILE_PATH = r"C:\Users\Anusri\OneDrive\Documents\Desktop\T_SCADA_DEMAND_BLOCK_6Months.xlsx"
SHEET     = "T_SCADA_DEMAND_BLOCK"

print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

df = pd.read_excel(FILE_PATH, sheet_name=SHEET)

if "Solar/Non-solar" in df.columns:
    df.drop(columns=["Solar/Non-solar"], inplace=True, errors="ignore")

df["SCH_DATE"] = pd.to_datetime(df["SCH_DATE"], errors="coerce")
df["Hour"]     = df["Hour"].astype(int)
df["DEMAND"]   = pd.to_numeric(df["DEMAND"], errors="coerce")
df["Month"]    = df["SCH_DATE"].dt.month
df["Day"]      = df["SCH_DATE"].dt.day

print(f"  Shape          : {df.shape}")
print(f"  Date range     : {df['SCH_DATE'].min()} → {df['SCH_DATE'].max()}")
print(f"  Missing DEMAND : {df['DEMAND'].isnull().sum()}")
print(f"  Duplicates     : {df.duplicated().sum()}")
print()

print("=" * 70)
print("2. FETCHING WEATHER DATA")
print("=" * 70)

latitude, longitude = 26.8467, 80.9462
start_date, end_date = "2025-01-01", "2025-06-10"

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
)

response   = requests.get(url)
weather_df = pd.DataFrame(response.json()["hourly"])

weather_df["time"]     = pd.to_datetime(weather_df["time"])
weather_df["SCH_DATE"] = weather_df["time"].dt.normalize()
weather_df["Hour"]     = weather_df["time"].dt.hour + 1
weather_df.drop(columns=["time"], inplace=True)

print(f"  Weather rows: {len(weather_df)}")

merged_df = pd.merge(df, weather_df, on=["SCH_DATE", "Hour"], how="inner")
merged_df.sort_values(["SCH_DATE", "Hour"], inplace=True)
merged_df.reset_index(drop=True, inplace=True)

merged_df["Month"] = merged_df["SCH_DATE"].dt.month

mask_null = merged_df["DEMAND"].isnull()
monthly_mean = merged_df.groupby("Month")["DEMAND"].transform("mean")
merged_df["DEMAND"] = merged_df["DEMAND"].fillna(monthly_mean)

print(f"  Merged rows   : {len(merged_df)}")
print(f"  Null after fill: {merged_df['DEMAND'].isnull().sum()}")
print()

print("=" * 70)
print("3. EXPLORATORY DATA ANALYSIS — VISUALIZATIONS")
print("=" * 70)

numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10, 7))
sns.heatmap(merged_df[numeric_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Overall Correlation Matrix")
plt.tight_layout()
plt.savefig("plot_01_correlation_matrix.png", bbox_inches="tight")
plt.close()

monthly_demand = merged_df.groupby(["Month", "Day"])["DEMAND"].mean().reset_index()
pivot_df = monthly_demand.pivot(index="Day", columns="Month", values="DEMAND")
pivot_df.columns = [calendar.month_abbr[m] for m in pivot_df.columns]

plt.figure(figsize=(12, 6))
pivot_df.plot(ax=plt.gca(), linewidth=2, marker=".")
plt.title("Electricity Demand Trend Over Days by Month")
plt.xlabel("Day of the Month")
plt.ylabel("Average Electricity Demand (MW)")
plt.legend(title="Month")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_02_monthly_demand_trend.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
sns.boxplot(x=merged_df["DEMAND"], color="salmon")
plt.title("Outlier Detection — Demand Distribution")
plt.xlabel("Demand (MW)")
plt.tight_layout()
plt.savefig("plot_03_demand_boxplot.png", bbox_inches="tight")
plt.close()

merged_df["DayOfWeek"]  = merged_df["SCH_DATE"].dt.dayofweek
merged_df["IsWeekend"]  = merged_df["DayOfWeek"].isin([5, 6]).astype(int)

weekend_avg = merged_df.groupby("IsWeekend")["DEMAND"].mean().reset_index()
weekend_avg["Label"] = weekend_avg["IsWeekend"].map({0: "Weekday", 1: "Weekend"})

plt.figure(figsize=(6, 4))
sns.barplot(x="Label", y="DEMAND", data=weekend_avg, palette="Set2")
plt.title("Average Demand: Weekday vs Weekend")
plt.ylabel("Average Demand (MW)")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("plot_04_weekday_vs_weekend.png", bbox_inches="tight")
plt.close()

hourly_trend = merged_df.groupby(["Hour", "IsWeekend"])["DEMAND"].mean().reset_index()
hourly_trend["DayType"] = hourly_trend["IsWeekend"].map({0: "Weekday", 1: "Weekend"})

plt.figure(figsize=(10, 5))
sns.lineplot(data=hourly_trend, x="Hour", y="DEMAND", hue="DayType", marker="o")
plt.title("Hourly Demand: Weekday vs Weekend")
plt.xlabel("Hour of Day")
plt.ylabel("Demand (MW)")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_05_hourly_weekday_weekend.png", bbox_inches="tight")
plt.close()

monthly_avg = merged_df.groupby("Month")["DEMAND"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(x="Month", y="DEMAND", data=monthly_avg, marker="o", linewidth=2, color="coral")
plt.title("Average Electricity Demand per Month")
plt.xlabel("Month")
plt.ylabel("Average Demand (MW)")
month_names = [calendar.month_name[m] for m in monthly_avg["Month"]]
plt.xticks(monthly_avg["Month"], month_names, rotation=30)
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_06_monthly_avg_demand.png", bbox_inches="tight")
plt.close()

merged_df["temp_bin"]     = (merged_df["temperature_2m"] // 2) * 2
merged_df["humidity_bin"] = (merged_df["relative_humidity_2m"] // 5) * 5
merged_df["wind_bin"]     = (merged_df["windspeed_10m"] // 2) * 2

temp_demand     = merged_df.groupby("temp_bin")["DEMAND"].mean().reset_index()
humidity_demand = merged_df.groupby("humidity_bin")["DEMAND"].mean().reset_index()
wind_demand     = merged_df.groupby("wind_bin")["DEMAND"].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x="temp_bin", y="DEMAND", data=temp_demand, color="salmon", ax=axes[0])
axes[0].set_title("Avg Demand vs Temperature")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Demand (MW)")
axes[0].tick_params(axis="x", rotation=45)

sns.barplot(x="humidity_bin", y="DEMAND", data=humidity_demand, color="skyblue", ax=axes[1])
axes[1].set_title("Avg Demand vs Humidity")
axes[1].set_xlabel("Humidity (%)")
axes[1].set_ylabel("")
axes[1].tick_params(axis="x", rotation=45)

sns.barplot(x="wind_bin", y="DEMAND", data=wind_demand, color="lightgreen", ax=axes[2])
axes[2].set_title("Avg Demand vs Windspeed")
axes[2].set_xlabel("Windspeed (m/s)")
axes[2].set_ylabel("")
axes[2].tick_params(axis="x", rotation=45)

plt.suptitle("Demand vs Weather Features", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot_07_demand_vs_weather.png", bbox_inches="tight")
plt.close()

corr_features = [
    "Hour", "temperature_2m", "relative_humidity_2m", "windspeed_10m",
    "DayOfWeek", "IsWeekend", "DEMAND",
]

unique_months = sorted(merged_df["Month"].unique())
n_months = len(unique_months)
fig, axes = plt.subplots(1, n_months, figsize=(5 * n_months, 4))
if n_months == 1:
    axes = [axes]

for ax, month in zip(axes, unique_months):
    month_data = merged_df[merged_df["Month"] == month][corr_features]
    sns.heatmap(month_data.corr(), annot=True, fmt=".2f", cmap="magma",
                ax=ax, cbar=False, linewidths=0.3)
    ax.set_title(f"{calendar.month_name[month]}")

plt.suptitle("Per-Month Correlation Heatmaps", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("plot_08_per_month_corr.png", bbox_inches="tight")
plt.close()

print("  ✓ All EDA plots saved.\n")

print("=" * 70)
print("4. FEATURE ENGINEERING")
print("=" * 70)

merged_df["Hour_sin"]  = np.sin(2 * np.pi * merged_df["Hour"] / 24)
merged_df["Hour_cos"]  = np.cos(2 * np.pi * merged_df["Hour"] / 24)
merged_df["Month_sin"] = np.sin(2 * np.pi * merged_df["Month"] / 12)
merged_df["Month_cos"] = np.cos(2 * np.pi * merged_df["Month"] / 12)

features = [
    "Hour_sin", "Hour_cos",
    "Month_sin", "Month_cos",
    "temperature_2m", "relative_humidity_2m", "windspeed_10m",
    "DayOfWeek", "IsWeekend",
]

X = merged_df[features].copy()
y = merged_df["DEMAND"].copy()

valid = X.notnull().all(axis=1) & y.notnull()
X, y = X[valid], y[valid]

print(f"  Features       : {features}")
print(f"  Samples used   : {len(X)}")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly  = poly.transform(X_test_scaled)

models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
    },
    "Polynomial Regression (deg=2)": {
        "model": LinearRegression(),
        "X_train": X_train_poly,
        "X_test":  X_test_poly,
    },
    "Ridge Regression": {
        "model": Ridge(alpha=1.0),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
    },
    "SVR (RBF Kernel)": {
        "model": SVR(kernel="rbf", C=100, epsilon=0.1),
        "X_train": X_train_scaled,
        "X_test":  X_test_scaled,
    },
    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        ),
        "X_train": X_train,
        "X_test":  X_test,
    },
    "XGBoost": {
        "model": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "X_train": X_train,
        "X_test":  X_test,
    },
    "LightGBM": {
        "model": LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "X_train": X_train,
        "X_test":  X_test,
    },
}

print("=" * 70)
print("5. TRAINING & EVALUATION")
print("=" * 70)

results = {}

for name, cfg in models.items():
    print(f"\n  ▸ Training {name} …", end=" ")

    model     = cfg["model"]
    Xtr, Xte  = cfg["X_train"], cfg["X_test"]

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "MAE":    mae,
        "RMSE":   rmse,
        "R2":     r2,
        "MAPE":   mape,
    }

    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  MAPE={mape:.2f}%")

print("\n" + "=" * 70)
print("6. MODEL COMPARISON SUMMARY")
print("=" * 70)

summary_df = pd.DataFrame({
    name: {"MAE": r["MAE"], "RMSE": r["RMSE"], "R²": r["R2"], "MAPE (%)": r["MAPE"]}
    for name, r in results.items()
}).T.sort_values("R²", ascending=False)

print(summary_df.to_string())
print()

best_model_name = summary_df.index[0]
print(f" Best model by R²: {best_model_name}  (R² = {summary_df.loc[best_model_name, 'R²']:.4f})")
print()

print("=" * 70)
print("7. COMPARISON VISUALIZATIONS")
print("=" * 70)

model_names = list(results.keys())
rmse_vals   = [results[n]["RMSE"] for n in model_names]
r2_vals     = [results[n]["R2"]   for n in model_names]
mae_vals    = [results[n]["MAE"]  for n in model_names]
mape_vals   = [results[n]["MAPE"] for n in model_names]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

colors = sns.color_palette("viridis", len(model_names))

bars1 = axes[0].barh(model_names, rmse_vals, color=colors)
axes[0].set_xlabel("RMSE (MW)")
axes[0].set_title("Model Comparison — RMSE (lower is better)")
axes[0].invert_yaxis()
for bar, val in zip(bars1, rmse_vals):
    axes[0].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=9)

bars2 = axes[1].barh(model_names, r2_vals, color=colors)
axes[1].set_xlabel("R² Score")
axes[1].set_title("Model Comparison — R² (higher is better)")
axes[1].invert_yaxis()
for bar, val in zip(bars2, r2_vals):
    axes[1].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot_09_model_comparison_rmse_r2.png", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

bars3 = axes[0].barh(model_names, mae_vals, color=sns.color_palette("magma", len(model_names)))
axes[0].set_xlabel("MAE (MW)")
axes[0].set_title("Model Comparison — MAE (lower is better)")
axes[0].invert_yaxis()
for bar, val in zip(bars3, mae_vals):
    axes[0].text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=9)

bars4 = axes[1].barh(model_names, mape_vals, color=sns.color_palette("magma", len(model_names)))
axes[1].set_xlabel("MAPE (%)")
axes[1].set_title("Model Comparison — MAPE (lower is better)")
axes[1].invert_yaxis()
for bar, val in zip(bars4, mape_vals):
    axes[1].text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("plot_10_model_comparison_mae_mape.png", bbox_inches="tight")
plt.close()

n_models = len(results)
cols = 3
rows = (n_models + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axes = axes.flatten()

for idx, (name, r) in enumerate(results.items()):
    ax = axes[idx]
    ax.scatter(y_test, r["y_pred"], alpha=0.3, s=10, color=colors[idx])
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            "r--", linewidth=1.5)
    ax.set_xlabel("Actual Demand (MW)")
    ax.set_ylabel("Predicted Demand (MW)")
    ax.set_title(f"{name}\nR²={r['R2']:.4f}")
    ax.grid(True, alpha=0.3)

for j in range(idx + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Actual vs Predicted — All Models", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("plot_11_actual_vs_predicted_scatter.png", bbox_inches="tight")
plt.close()

top3 = summary_df.index[:3].tolist()

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
sample_range = slice(0, 200)

y_test_arr = y_test.values

for ax, name in zip(axes, top3):
    preds = results[name]["y_pred"]
    ax.plot(y_test_arr[sample_range], label="Actual", color="steelblue", linewidth=1.5)
    ax.plot(preds[sample_range],      label="Predicted", color="coral", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Demand (MW)")
    ax.set_title(f"{name}  (R²={results[name]['R2']:.4f})")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Sample Index")
plt.suptitle("Actual vs Predicted (First 200 Test Samples) — Top 3 Models", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plot_12_actual_vs_predicted_line.png", bbox_inches="tight")
plt.close()

tree_models = ["Random Forest", "XGBoost", "LightGBM"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, name in zip(axes, tree_models):
    importances = results[name]["model"].feature_importances_
    sorted_idx  = np.argsort(importances)
    ax.barh([features[i] for i in sorted_idx], importances[sorted_idx],
            color=sns.color_palette("viridis", len(features)))
    ax.set_title(f"Feature Importance — {name}")
    ax.set_xlabel("Importance")

plt.tight_layout()
plt.savefig("plot_13_feature_importance.png", bbox_inches="tight")
plt.close()

best_preds    = results[best_model_name]["y_pred"]
best_residuals = y_test.values - best_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(best_residuals, bins=50, color="teal", edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5)
axes[0].set_title(f"Residual Distribution — {best_model_name}")
axes[0].set_xlabel("Residual (Actual − Predicted)")
axes[0].set_ylabel("Frequency")

axes[1].scatter(best_preds, best_residuals, alpha=0.3, s=10, color="teal")
axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
axes[1].set_title(f"Residuals vs Predicted — {best_model_name}")
axes[1].set_xlabel("Predicted Demand (MW)")
axes[1].set_ylabel("Residual")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_14_residuals_best_model.png", bbox_inches="tight")
plt.close()

print("=" * 70)
print("8. DEMAND vs PREDICTED SUPPLY GAP ANALYSIS")
print("=" * 70)

gap = y_test.values - best_preds
print(f"  Mean gap (Actual − Predicted)   : {np.mean(gap):.2f} MW")
print(f"  Std-dev of gap                  : {np.std(gap):.2f} MW")
print(f"  Max under-prediction            : {np.max(gap):.2f} MW")
print(f"  Max over-prediction             : {np.min(gap):.2f} MW")

plt.figure(figsize=(12, 5))
plt.plot(gap[:300], color="purple", alpha=0.7, linewidth=1)
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.fill_between(range(len(gap[:300])), gap[:300], 0,
                 where=(gap[:300] > 0), color="orangered", alpha=0.3, label="Under-supplied")
plt.fill_between(range(len(gap[:300])), gap[:300], 0,
                 where=(gap[:300] < 0), color="steelblue", alpha=0.3, label="Over-supplied")
plt.title(f"Demand–Supply Gap (First 300 Samples) — {best_model_name}")
plt.xlabel("Sample Index")
plt.ylabel("Gap (MW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_15_demand_supply_gap.png", bbox_inches="tight")
plt.close()

print()
print("=" * 70)
print(" ALL DONE — Plots saved as plot_01 … plot_15 in the working directory.")
print("=" * 70)
print()
print(summary_df.to_string())
print()
print(f"Best model: {best_model_name}  |  R² = {summary_df.loc[best_model_name, 'R²']:.4f}"
      f"  |  RMSE = {summary_df.loc[best_model_name, 'RMSE']:.2f}")

print("\n" + "=" * 70)
print(f"SAMPLE PREDICTIONS (Best Model: {best_model_name})")
print("=" * 70)
sample_compare = pd.DataFrame({
    "Actual Demand (MW)": y_test.values[:15],
    "Predicted Demand (MW)": best_preds[:15].round(2),
    "Difference (MW)": (y_test.values[:15] - best_preds[:15]).round(2)
})
print(sample_compare.to_string())
print("=" * 70 + "\n")
