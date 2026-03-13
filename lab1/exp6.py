import pandas as pd
df = pd.read_csv("data.csv")
print("First 5 Records:")
print(df.head())
print("\nLast 5 Records:")
print(df.tail())