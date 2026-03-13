import pandas as pd
data = {
    "Name": ["Alice", "Bob", "Charlie", "David"],
    "Marks": [85, 90, 78, 92]
}
df = pd.DataFrame(data)
print("Student DataFrame:")
print(df)
print("\nData Types:")
print(df.dtypes)