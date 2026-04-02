import pandas as pd
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/rashida048/Datasets/master/StudentsPerformance.csv"
df = pd.read_csv(url)
num_cols = df.select_dtypes(include=['int64','float64']).columns
print("Showing Histograms...")
df[num_cols].hist(figsize=(10,6))
plt.show()
print("Showing Boxplot...")
df[num_cols].plot(kind='box', figsize=(8,6))
plt.show()
