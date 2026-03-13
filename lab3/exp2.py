import pandas as pd
url = "https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv"
df = pd.read_csv(url)
num_cols = df.select_dtypes(include=['int64','float64']).columns
Q1 = df[num_cols].quantile(0.25)
Q2 = df[num_cols].quantile(0.50)
Q3 = df[num_cols].quantile(0.75)

print("First Quartile (Q1):\n", Q1)
print("\nSecond Quartile (Median Q2):\n", Q2)
print("\nThird Quartile (Q3):\n", Q3)
