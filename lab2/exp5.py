import pandas as pd
df = pd.read_csv("data.csv")

print("Original Columns:")
print(df.columns)
df.rename(columns={
    "Name": "Student_Name",
    "Marks": "Student_Marks"
}, inplace=True)

print("\nUpdated DataFrame:")
print(df)