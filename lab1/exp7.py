import pandas as pd
df = pd.read_csv("data.csv")
print("Dataset Info:")
print(df.info())
print("\nShape of dataset:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())