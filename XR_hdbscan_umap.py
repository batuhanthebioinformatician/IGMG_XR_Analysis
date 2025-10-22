# filename: XR_hdbscan_umap.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
import umap.umap_ as umap
from scipy.spatial.distance import squareform
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting

# -----------------------------------------------
# User settings
# -----------------------------------------------
# Path to the distance matrix (square, symmetric; zeros on diagonal).
excel_path = r"/path/to/XR_BLOSUM45_distance.xlsx"
# Output artifact paths.
out_csv = "XR_HDBSCAN_exploration_clusters.csv"
umap_png = "XR_HDBSCAN_exploration_UMAP.png"
umap3d_png = "XR_HDBSCAN_exploration_UMAP_3D.png"
tree_png = "XR_HDBSCAN_CondensedTree.png"

# HDBSCAN parameters
min_cluster_size = 7
min_samples = 5
epsilon = 0.0

# UMAP parameters
n_neighbors = 15
min_dist = 0.1

# -----------------------------------------------
# Load distance matrix
# -----------------------------------------------
df_dist = pd.read_excel(excel_path, index_col=0)
assert np.allclose(df_dist.values, df_dist.values.T), "Matrix must be symmetric"
assert np.all(np.diag(df_dist.values) == 0), "Diagonal must be zero"
distance_matrix = df_dist.values.astype(np.float64)

# -----------------------------------------------
# HDBSCAN clustering
# -----------------------------------------------
clusterer = hdbscan.HDBSCAN(
    metric="precomputed",
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    cluster_selection_epsilon=epsilon,
)
labels = clusterer.fit_predict(distance_matrix)

df_clusters = pd.DataFrame({"Sequence_ID": df_dist.index, "Cluster": labels})
df_clusters.to_csv(out_csv, index=False)
print(f"Saved cluster assignments → {out_csv}")
print(f"Clusters (excluding -1): {len(set(labels) - {-1})}")
print(f"Noise points (-1): {np.sum(labels == -1)}")

# -----------------------------------------------
# Condensed tree visualization
# -----------------------------------------------
plt.figure(figsize=(12, 6))
clusterer.condensed_tree_.plot(
    select_clusters=True,
    selection_palette=sns.color_palette("husl", 20),
)
plt.title("HDBSCAN Condensed Tree")
plt.tight_layout()
plt.savefig(tree_png, dpi=300)
plt.close()
print(f"Condensed tree plot saved → {tree_png}")

# -----------------------------------------------
# UMAP 2D
# -----------------------------------------------
umap_2d = umap.UMAP(
    metric="precomputed",
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    random_state=42,
)
embedding_2d = umap_2d.fit_transform(distance_matrix)

# Consistent colors across plots
unique_labels = sorted(np.unique(labels))
cmap = cm.get_cmap("tab20", len(unique_labels))
color_dict = {lab: cmap(i) for i, lab in enumerate(unique_labels)}
label_colors = [color_dict[l] for l in labels]

plt.figure(figsize=(10, 6))
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=label_colors,
    s=60,
    edgecolors="w",
    linewidth=0.5,
)
handles = [
    mpatches.Patch(color=color_dict[l], label=f"Cluster {l}") for l in unique_labels
]
plt.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title(f"UMAP-2D (min_cluster_size={min_cluster_size}, min_samples={min_samples}, ε={epsilon})")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(umap_png, dpi=300)
plt.close()
print(f"2D UMAP saved → {umap_png}")

# -----------------------------------------------
# UMAP 3D
# -----------------------------------------------
umap_3d = umap.UMAP(
    n_components=3,
    metric="precomputed",
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    random_state=42,
)
embedding_3d = umap_3d.fit_transform(distance_matrix)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    embedding_3d[:, 0],
    embedding_3d[:, 1],
    embedding_3d[:, 2],
    c=label_colors,
    s=60,
    edgecolors="w",
    linewidth=0.5,
)
handles = [
    mpatches.Patch(color=color_dict[l], label=f"Cluster {l}") for l in unique_labels
]
ax.legend(handles=handles, title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
ax.set_title("3D UMAP of BLOSUM45 Distance Matrix (HDBSCAN clusters)")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_zlabel("UMAP-3")
plt.tight_layout()
plt.savefig(umap3d_png, dpi=300)
plt.close()
print(f"3D UMAP saved → {umap3d_png}")
