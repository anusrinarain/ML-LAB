import pandas as pd
url = "https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv"
df = pd.read_csv(url)
num_cols = df.select_dtypes(include=['int64','float64']).columns
corr_matrix = df[num_cols].corr()
print("Correlation Matrix:\n")
print(corr_matrix)
cov_matrix = df[num_cols].cov()
print("\nCovariance Matrix:\n")
print(cov_matrix)
