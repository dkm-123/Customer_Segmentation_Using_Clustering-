import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#Load Dataset
df = pd.read_csv("Customer_clean.csv")
pd.set_option('display.max_columns', None)

print("\nShape of dataset:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe())

#Missing Values
plt.figure(figsize=(10, 5))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

missing_percent = df.isnull().sum() / len(df) * 100
print("\nMissing Values Percentage:\n", missing_percent[missing_percent > 0])

#Duplicates
print("\nNumber of duplicate rows:", df.duplicated().sum())
# Optionally drop duplicates
# df = df.drop_duplicates()

#Numeric Columns Analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns

#Histograms
df[numeric_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Histograms for Numeric Columns")
plt.show()

#Boxplots & Outliers
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col], color='lightgreen')
    plt.title(f"Boxplot - {col}")
    plt.show()

    #Outlier detection using z-score
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = np.where(z_scores > 3)
    print(f"Number of outliers in {col}: {len(outliers[0])}")

#Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

#Categorical Columns Analysis
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10], palette='viridis')
    plt.title(f"Top 10 Categories - {col}")
    plt.show()

#Time-based Trends (if date exists)
if 'InvoiceDate' in df.columns:
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    #Monthly Quantity Sold
    monthly_quantity = df.groupby('InvoiceMonth')['Quantity'].sum()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=monthly_quantity.index, y=monthly_quantity.values, marker='o')
    plt.xticks(rotation=45)
    plt.title("Monthly Quantity Sold")
    plt.xlabel("Month")
    plt.ylabel("Total Quantity")
    plt.show()

    #Monthly Revenue (if UnitPrice exists)
    if 'UnitPrice' in df.columns:
        df['Revenue'] = df['Quantity'] * df['UnitPrice']
        monthly_revenue = df.groupby('InvoiceMonth')['Revenue'].sum()
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=monthly_revenue.index, y=monthly_revenue.values, marker='o')
        plt.xticks(rotation=45)
        plt.title("Monthly Revenue")
        plt.xlabel("Month")
        plt.ylabel("Total Revenue")
        plt.show()

print("\nAutomated EDA Completed ✅")
