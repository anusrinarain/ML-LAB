import pandas as pd

df = pd.read_csv("data.csv")

print("Original Data:")
print(df)
df.replace({"Male": "M", "Female": "F"}, inplace=True)

print("\nCorrected Data:")
print(df)