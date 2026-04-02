import pandas as pd
df = pd.read_csv("data.csv")
print("Statistical Summary:")
print(df.describe())

print("\nInterpretation:")
print("Mean: Average value of the column")
print("Std: Standard deviation (spread of data)")
print("Min: Minimum value")
print("Max: Maximum value")