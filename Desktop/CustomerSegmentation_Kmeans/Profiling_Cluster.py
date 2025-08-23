import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

#Setup Plot Saving Directory
plot_dir = "cluster_charts"
os.makedirs(plot_dir, exist_ok=True)  # Create folder if not exists

#Load clustered dataset
df = pd.read_csv(r"C:\Users\hp\Desktop\CustomerSegmentation_Kmeans\Customer_Segmentation_Using_Clustering-\customer_clusters.csv")

print("First 5 rows of dataset:")
print(df.head())
print("\nCluster distribution:")
print(df["Cluster"].value_counts())

#Average TotalSpent (Revenue) per Cluster
if "TotalSpent" in df.columns:
    avg_spent = df.groupby("Cluster")["TotalSpent"].mean()
    avg_spent.plot(kind="bar", color="pink", edgecolor="magenta")
    plt.title("Average Spending per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Avg TotalSpent")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "avg_spending_per_cluster.png"))
    plt.show()
else:
    print("⚠️ Column 'TotalSpent' not found in dataset")

#Order Quantities across Clusters
if "TotalQuantity" in df.columns:
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Cluster", y="TotalQuantity", data=df, palette="Set2")
    plt.title("Total Quantities across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Total Quantity")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "quantities_across_clusters.png"))
    plt.show()
else:
    print("⚠️ Column 'TotalQuantity' not found in dataset")

#Share of Customers in Each Cluster
plt.figure(figsize=(6,6))
cluster_counts = df["Cluster"].value_counts()
cluster_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90, cmap="Set3")
plt.title("Customer Share by Cluster")
plt.ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "customer_share_pie.png"))
plt.show()

#Summary Statistics per Cluster
summary = df.groupby("Cluster").agg({
    "CustomerID": "count",
    "NumTransactions": "mean",
    "TotalQuantity": "mean",
    "TotalSpent": "mean"
}).rename(columns={"CustomerID": "NumCustomers"})

print("\nCluster Summary Statistics:")
print(summary)

#Customer Lifetime Value (CLV)-like Metric
if "TotalSpent" in df.columns and "NumTransactions" in df.columns:
    df["CLV"] = df["NumTransactions"] * df["TotalSpent"]
    clv_cluster = df.groupby("Cluster")["CLV"].mean()
    clv_cluster.plot(kind="bar", color="beige", edgecolor="brown")
    plt.title("Average Customer Lifetime Value (CLV) per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Avg CLV")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "clv_per_cluster.png"))
    plt.show()

#RFM-style Summary (Frequency & Monetary)
if "TotalSpent" in df.columns and "NumTransactions" in df.columns:
    rfm_summary = df.groupby("Cluster").agg({
        "NumTransactions": "mean",
        "TotalSpent": "mean"
    }).rename(columns={
        "NumTransactions": "AvgFrequency",
        "TotalSpent": "AvgMonetary"
    })
    print("\nRFM-style Summary per Cluster:")
    print(rfm_summary)

#Radar/Spider Chart for Cluster Profiles
features = ["NumTransactions", "TotalQuantity", "TotalSpent"]
cluster_profile = df.groupby("Cluster")[features].mean()

# Normalize for comparison (0–1 scaling)
data = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
labels = data.columns
num_vars = len(labels)

plt.figure(figsize=(6,6))
for idx, row in data.iterrows():
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    values = row.values.flatten().tolist()
    values += values[:1]  # repeat first value to close loop
    angles += angles[:1]
    plt.polar(angles, values, label=f"Cluster {idx}")

plt.xticks([n / float(num_vars) * 2 * pi for n in range(num_vars)], labels)
plt.title("Cluster Profiles (Radar Chart)")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "cluster_profiling_radar.png"))
plt.show()

#Outlier Detection (Small Clusters)
min_cluster_size = df["Cluster"].value_counts().min()
small_clusters = df["Cluster"].value_counts()[df["Cluster"].value_counts() == min_cluster_size].index

for c in small_clusters:
    print(f"\nOutlier Cluster Detected: {c}")
    print(df[df["Cluster"] == c])

#Business Recommendations (Textual Insights)
print("\n--- Business Recommendations ---")
print("Cluster 0 → Large group, mid-level spenders → Good for mass marketing campaigns")
print("Cluster 1 → Smaller, low-spending → Target with promotions/discounts to boost activity")
print("Cluster 2 → High-spending customers → Prioritize with loyalty programs or premium offers")
print("Cluster 3 → Very small/outlier group → Investigate: could be data errors or niche segment")
