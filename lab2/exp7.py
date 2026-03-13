import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
plt.figure()
plt.plot(df["Year"], df["Sales"])
plt.title("Sales Trend Over Time")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.show()

plt.figure()
df["Category"].value_counts().plot(kind="bar")
plt.title("Category Comparison")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()