import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
plt.scatter(df["Age"], df["Marks"])

plt.title("Age vs Marks")
plt.xlabel("Age")
plt.ylabel("Marks")

plt.show()