import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
plt.hist(df["Marks"], bins=10)

plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")

plt.show()