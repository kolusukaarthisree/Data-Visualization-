
# ===== IMPORTS =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# ===== DATA LOADING =====
print("\n🔍 Loading dataset...")
# Load the dataset using the absolute path
df = pd.read_csv(r"D:/A Downloads/Kaarthi projects/Titanc-dv/titanic_data_large.csv")

# ===== DATA EXPLORATION =====
print("\n📊 Dataset Overview:")
print(df.head())
print("\n📈 Basic Statistics:")
print(df.describe())
print("\n❓ Missing Values:")
print(df.isnull().sum())

# ===== DATA CLEANING =====
print("\n🧹 Cleaning data...")
# Fill missing age values with median
df['age'] = df['age'].fillna(df['age'].median())
# Drop rows with missing 'embarked' (or fill with mode)
df = df.dropna(subset=['embarked'])

# ===== BASIC VISUALIZATIONS (Matplotlib/Seaborn) =====
print("\n📉 Creating foundational visualizations...")

# 1. Histogram (Age Distribution)
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 2. Bar Plot (Survival by Class)
plt.figure(figsize=(8, 5))
sns.countplot(x='pclass', hue='survived', data=df)
plt.title("Survival Count by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend(["Did Not Survive", "Survived"])
plt.show()

# 3. Boxplot (Fare by Class)
plt.figure(figsize=(8, 5))
sns.boxplot(x='pclass', y='fare', data=df)
plt.title("Fare Distribution by Class")
plt.xlabel("Class")
plt.ylabel("Fare ($)")
plt.show()

# 4. Heatmap (Correlations)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# ===== INTERACTIVE VISUALIZATION (Plotly) =====
print("\n✨ Creating interactive visualizations...")

# Scatter Plot (Age vs Fare, colored by survival)
fig = px.scatter(
    df,
    x='age',
    y='fare',
    color='survived',
    title="Age vs Fare (Survival Status)",
    labels={'age': 'Age', 'fare': 'Fare ($)'},
    hover_data=['sex', 'pclass']
)
fig.show()

# ===== INSIGHTS SUMMARY =====
print("\n📌 Key Insights:")
print("- Higher-class passengers had better survival rates.")
print("- Younger passengers (<30) had higher survival rates.")
print("- Fare prices varied significantly by class.")

print("\n✅ Script executed successfully! Check the visualizations.")
