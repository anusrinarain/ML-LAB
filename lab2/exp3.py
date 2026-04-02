import pandas as pd
df = pd.read_csv("data.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)

print("Dataset after replacing missing values:")
print(df)