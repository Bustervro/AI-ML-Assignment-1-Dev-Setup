# ==============================================
# Iris Flower Dataset Exploration
# Aden Osman
# 10/29/25
# ==============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


iris = load_iris()

# Create pandas DataFrame
df = pd.DataFrame(
    data=iris.data,
    columns=iris.feature_names
)

df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)


print("First 5 Rows of the Dataset:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

print("\nUnique Species:")
print(df["species"].unique())

# --- Visualizing Data ---

# 1. Histogram
plt.figure(figsize=(6,4))
sns.histplot(df["petal length (cm)"], bins=20, kde=True, color='skyblue')
plt.title("Histogram of Petal Length")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 2. Box plot organized by species
plt.figure(figsize=(6,4))
sns.boxplot(x="species", y="sepal width (cm)", data=df, palette="Set2")
plt.title("Box Plot of Sepal Width by Species")
plt.xlabel("Species")
plt.ylabel("Sepal Width (cm)")
plt.show()

# 3. The Scatter plot (Length vs. Width)
plt.figure(figsize=(6,5))
sns.scatterplot(
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    data=df,
    palette="viridis",
    s=80
)
plt.title("Petal Length vs. Petal Width by Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.legend(title="Species")
plt.show()

# ---Findings ---
print("\n--- Key Findings ---")
print("1. Petal length and width show clear separation between species, especially Setosa vs. others.")
print("2. Sepal features overlap more, making them less useful for species separation.")
print("3. No missing values are found, and all features are numeric.")
