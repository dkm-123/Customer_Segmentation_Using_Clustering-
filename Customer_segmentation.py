# customer_segmentation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv(r"C:\Users\Admin\Customer_Segmentation_Using_Clustering-\Customer_clean.csv")

# Fix InvoiceDate format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates (if any)
df = df.dropna(subset=['InvoiceDate'])

# ==========================
# 2. Feature Engineering
# ==========================
# Create 'TotalAmount'
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Group by CustomerID to create RFM-like features
customer_df = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',          # Number of unique transactions
    'Quantity': 'sum',               # Total items bought
    'TotalAmount': 'sum',            # Total money spent
}).reset_index()

customer_df.rename(columns={
    'InvoiceNo': 'NumTransactions',
    'Quantity': 'TotalQuantity',
    'TotalAmount': 'TotalSpent'
}, inplace=True)

# ==========================
# 3. Preprocessing & Scaling
# ==========================
features = ['NumTransactions', 'TotalQuantity', 'TotalSpent']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df[features])

# ==========================
# 4. Find Optimal K
# ==========================
inertia = []
silhouette_scores = []
scores = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    inertia.append(kmeans.inertia_)
    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)

    silhouette_scores.append(silhouette)
    scores.append([k, silhouette, calinski, davies])

# Save scores to CSV
scores_df = pd.DataFrame(scores, columns=["K", "Silhouette", "Calinski_Harabasz", "Davies_Bouldin"])
scores_df.to_csv("clustering_scores.csv", index=False)

# Plot Elbow Method
plt.plot(K_range, inertia, 'bx-')
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.savefig("elbow_method.png")
plt.close()

# ==========================
# 5. Apply KMeans (choose K=4 as example)
# ==========================
kmeans = KMeans(n_clusters=4, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Save customer segmentation results
customer_df.to_csv("customer_clusters.csv", index=False)

# ==========================
# 6. Cluster Summary
# ==========================
cluster_summary = customer_df.groupby('Cluster')[features].mean()
cluster_summary.to_csv("cluster_summary.csv")

# ==========================
# 7. PCA Visualization
# ==========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                hue=customer_df['Cluster'], palette="Set2", s=50)
plt.title("Customer Segments (PCA Visualization)")
plt.savefig("pca_clusters.png")
plt.close()

print("✅ All outputs saved: customer_clusters.csv, cluster_summary.csv, clustering_scores.csv, elbow_method.png, pca_clusters.png")
