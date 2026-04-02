import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data.csv")
plt.boxplot(df["Marks"])

plt.title("Boxplot of Marks")

plt.show()