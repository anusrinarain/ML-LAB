import pandas as pd
df = pd.read_csv("data.csv")

print("Original dataset size:", df.shape)
cleaned_df = df.dropna()

print("Dataset size after removing missing values:", cleaned_df.shape)

print("\nCleaned Dataset:")
print(cleaned_df)