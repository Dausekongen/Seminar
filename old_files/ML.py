import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
data = pd.read_csv("ml_dataset.csv", index_col=0, parse_dates=True)

# -----------------------------
# 2️⃣ Features for clustering
# -----------------------------
features = [
    "Inflation",
    "Inflation_mom",
    "IP_growth",
    "Yield_curve",
    "Rate_level",
    "Unemployment",
    "Unemployment_change",
    "Infl_exp",
    "SP500_vol",
    "Bond_vol"
]

# Drop missing values
data = data.dropna(subset=features)

# -----------------------------
# 3️⃣ Feature matrix
# -----------------------------
X = data[features]

# -----------------------------
# 4️⃣ Scale Features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 5️⃣ K-Means Clustering (dynamic k)
# -----------------------------
k = 7  # CHANGE THIS TO NUMBER OF CLUSTERS YOU WANT
kmeans = KMeans(n_clusters=k, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster counts:")
print(data["Cluster"].value_counts())

# -----------------------------
# 6️⃣ PCA (dynamic to 3D or number of clusters)
# -----------------------------
n_pca = min(3, k)  # for 3D plotting, max 3 components
pca = PCA(n_components=n_pca)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained variance ratio:", pca.explained_variance_ratio_)

pc_labels = [f"PC{i+1}" for i in range(n_pca)]
print("\nPCA components:")
print(pd.DataFrame(pca.components_[:, :], columns=features, index=pc_labels))

# -----------------------------
# 7️⃣ 3D PCA Scatter Plot
# -----------------------------
if n_pca == 3:
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab10.colors

    for cluster in range(k):
        mask = data["Cluster"] == cluster
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            X_pca[mask, 2],
            color=colors[cluster % len(colors)],
            s=40,
            label=f"Cluster {cluster}"
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("K-Means Clusters in 3D PCA Space")
    ax.legend()
    plt.show()

# -----------------------------
# 8️⃣ Plot Inflation over time with clusters
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(data.index, data["Inflation"], color='gray', alpha=0.5)

colors = plt.cm.tab10.colors
for cluster in range(k):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(
        cluster_data.index,
        cluster_data["Inflation"],
        color=colors[cluster % len(colors)],
        s=20,
        label=f"Cluster {cluster}"
    )

plt.title("Inflation Over Time (Clusters)")
plt.xlabel("Date")
plt.ylabel("Inflation")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# 9️⃣ Plot S&P500 over time with clusters
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(data.index, data["SP500"], color='gray', alpha=0.5)

for cluster in range(k):
    cluster_data = data[data["Cluster"] == cluster]
    plt.scatter(
        cluster_data.index,
        cluster_data["SP500"],
        color=colors[cluster % len(colors)],
        s=20,
        label=f"Cluster {cluster}"
    )

plt.title("S&P 500 Over Time (Clusters)")
plt.xlabel("Date")
plt.ylabel("S&P 500")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# 🔟 Cluster profiles (means)
# -----------------------------
cluster_means = data.groupby("Cluster")[features].mean()
print("\nAverage values per cluster:")
print(cluster_means)

# Bar plot
cluster_means.plot(kind="bar", figsize=(14,6))
plt.title("Average Feature Values per Cluster")
plt.ylabel("Mean")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(cluster_means, annot=True, fmt=".3f", cmap="coolwarm")
plt.title("Cluster Profiles")
plt.xlabel("Features")
plt.ylabel("Cluster")
plt.show()