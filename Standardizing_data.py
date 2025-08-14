#Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Load dataset
df = pd.read_csv("Customer_clean.csv")

#Drop 'Description' (high-cardinality text)
df = df.drop(columns=['Description'], errors='ignore')

#One-Hot Encode 'Country'
if 'Country' in df.columns:
 df = pd.get_dummies(df, columns=['Country'], drop_first=True)

#Selecting numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

#Initialize Standard scaler
scaler = StandardScaler()

#Fit and transform numeric columns 
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#Check result
print("First 5 rows after preprocessing:\n", df_scaled.head())
print("\nSummary statistics for numeric columns:\n", df_scaled[numeric_cols].describe())

#Save preprocessed dataset
df_scaled.to_csv("Customer_preprocessed.csv", index=False)

import pandas as pd

#Load preprocessed dataset
df_scaled = pd.read_csv("Customer_preprocessed.csv")

#Convert InvoiceDate to datetime
df_scaled['InvoiceDate'] = pd.to_datetime(df_scaled['InvoiceDate'], dayfirst=True, errors='coerce')

#Extract Month and Year
df_scaled['InvoiceMonth'] = df_scaled['InvoiceDate'].dt.month
df_scaled['InvoiceYear'] = df_scaled['InvoiceDate'].dt.year

#Drop the original InvoiceDate column
df_scaled = df_scaled.drop(columns=['InvoiceDate'])

#Check result
print(df_scaled[['InvoiceMonth', 'InvoiceYear']].head())

from sklearn.preprocessing import StandardScaler

#Select all numeric columns
numeric_cols = df_scaled.select_dtypes(include=['int64', 'float64', 'float32', 'float64']).columns

#Initialize scaler
scaler = StandardScaler()

#Fit and transform numeric columns including InvoiceMonth and InvoiceYear
df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

#Check result
print(df_scaled[['InvoiceMonth', 'InvoiceYear']].head())
print(df_scaled[numeric_cols].describe())

