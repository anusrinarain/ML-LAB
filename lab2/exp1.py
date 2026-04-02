import pandas as pd
df = pd.read_csv("data.csv")
print("Dataset:")
print(df)
print("\nMissing values using isnull():")
print(df.isnull())

print("\nNon-missing values using notnull():")
print(df.notnull())
print("\nTotal missing values in each column:")
print(df.isnull().sum())