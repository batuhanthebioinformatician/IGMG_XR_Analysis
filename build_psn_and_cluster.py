# filename: build_psn_and_cluster.py
"""
Builds a Protein Similarity Network (PSN) from a precomputed BLOSUM45 similarity
matrix (CSV). Each node corresponds to a protein, and weighted edges represent
pairwise sequence similarity. The script constructs either a k-nearest-neighbor
or threshold-based network, applies community detection (Leiden or Louvain), and
exports cluster assignments and network files for downstream visualization or
comparison with reference classifications.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

# ------------------------------ PARAMETERS ------------------------------

# Input similarity matrix (square CSV; rows/cols = sequence IDs)
sim_csv = "XR_BLOSUM45_similarity.csv"

# Graph construction mode
use_knn = True          # True → use k-nearest-neighbor graph, False → use threshold-based graph
k_value = 10            # number of neighbors if k-NN is used
threshold_value = 25000 # similarity threshold if threshold-based mode is used

# Clustering parameters
cluster_method = "auto"  # "leiden", "louvain", or "auto"
resolution = 1.0
seed = 42

# Optional reference labels (for evaluation)
ref_labels_csv = None   # e.g. "reference_labels.csv"

# Output prefix
out_prefix = "XR_PSN"

# ------------------------------------------------------------------------

try:
    import igraph as ig
    import leidenalg as la
    HAVE_LEIDEN = True
except Exception:
    HAVE_LEIDEN = False

try:
    import community as community_louvain
    HAVE_LOUVAIN = True
except Exception:
    HAVE_LOUVAIN = False

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    from scipy.optimize import linear_sum_assignment
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


# ------------------------------ GRAPH BUILDING ------------------------------

def build_knn_psn(ids, sim, k):
    df = pd.DataFrame(sim, index=ids, columns=ids)
    G = nx.Graph()
    G.add_nodes_from(ids)
    for u in ids:
        nbrs = df.loc[u].drop(u).nlargest(k)
        for v, w in nbrs.items():
            w = float(w)
            if G.has_edge(u, v):
                if w > G[u][v]["weight"]:
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)
    return G


def build_threshold_psn(ids, sim, tau):
    n = len(ids)
    G = nx.Graph()
    G.add_nodes_from(ids)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w >= tau:
                G.add_edge(ids[i], ids[j], weight=w)
    return G


# ------------------------------ CLUSTERING ------------------------------

def cluster_graph_leiden(G, resolution=1.0, seed=42):
    if not HAVE_LEIDEN:
        raise RuntimeError("Leiden algorithm not available.")
    mapping = {node: idx for idx, node in enumerate(G.nodes())}
    inv_map = {v: k for k, v in mapping.items()}
    edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    g = ig.Graph(n=len(G), edges=edges, directed=False)
    g.es["weight"] = weights
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed
    )
    return {inv_map[i]: int(c) for i, c in enumerate(part.membership)}


def cluster_graph_louvain(G, resolution=1.0, seed=42):
    if not HAVE_LOUVAIN:
        raise RuntimeError("Louvain algorithm not available.")
    import numpy as np
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        part = community_louvain.best_partition(
            G, weight="weight", resolution=resolution, random_state=seed
        )
    finally:
        np.random.set_state(rng_state)
    return {str(k): int(v) for k, v in part.items()}


# ------------------------------ LABEL ALIGNMENT ------------------------------

def align_labels_to_reference(df_nodes, ref_col, psn_col):
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn and scipy are required for label alignment.")
    C = contingency_matrix(df_nodes[ref_col], df_nodes[psn_col])
    r_ind, c_ind = linear_sum_assignment(C.max() - C)
    psn_unique = sorted(df_nodes[psn_col].unique())
    ref_unique = sorted(df_nodes[ref_col].unique())
    mapping = {psn_unique[c]: ref_unique[r] for r, c in zip(r_ind, c_ind)}
    df_nodes["cluster_aligned"] = df_nodes[psn_col].map(mapping)
    ari = adjusted_rand_score(df_nodes[ref_col], df_nodes[psn_col])
    nmi = normalized_mutual_info_score(df_nodes[ref_col], df_nodes[psn_col])
    cm = pd.crosstab(df_nodes[ref_col], df_nodes[psn_col])
    return mapping, ari, nmi, cm


# ------------------------------ MAIN ------------------------------

def main():
    df_sim = pd.read_csv(sim_csv, index_col=0)
    ids = df_sim.index.tolist()
    sim = df_sim.values.astype(np.float64)
    print(f"Loaded similarity matrix ({df_sim.shape[0]}×{df_sim.shape[1]})")

    if use_knn:
        G = build_knn_psn(ids, sim, k_value)
        graph_params = {"mode": "knn", "k": k_value}
    else:
        G = build_threshold_psn(ids, sim, threshold_value)
        graph_params = {"mode": "threshold", "tau": threshold_value}

    if cluster_method == "auto":
        method = "leiden" if HAVE_LEIDEN else ("louvain" if HAVE_LOUVAIN else None)
        if method is None:
            raise RuntimeError("No clustering backend available.")
    else:
        method = cluster_method

    if method == "leiden":
        labels = cluster_graph_leiden(G, resolution, seed)
    else:
        labels = cluster_graph_louvain(G, resolution, seed)

    df_nodes = pd.DataFrame({"Sequence_ID": list(G.nodes())})
    df_nodes["psn_cluster"] = df_nodes["Sequence_ID"].map(labels).astype(int)
    df_nodes = df_nodes.sort_values(["psn_cluster", "Sequence_ID"])
    df_nodes.to_csv(f"{out_prefix}_nodes.csv", index=False)

    edges = [(u, v, float(d.get("weight", 1.0))) for u, v, d in G.edges(data=True)]
    pd.DataFrame(edges, columns=["source", "target", "weight"]).to_csv(f"{out_prefix}_edges.csv", index=False)

    nx.write_gexf(G, f"{out_prefix}.gexf")

    params = {
        "source": {"type": "similarity_csv", "path": str(sim_csv)},
        "graph_params": graph_params,
        "clustering": {"method": method, "resolution": resolution, "seed": seed},
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
    }
    with open(f"{out_prefix}_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    if ref_labels_csv:
        ref = pd.read_csv(ref_labels_csv)
        if not {"Sequence_ID", "ReferenceLabel"}.issubset(ref.columns):
            raise ValueError("Reference CSV must include 'Sequence_ID' and 'ReferenceLabel'.")
        merged = pd.merge(df_nodes, ref[["Sequence_ID", "ReferenceLabel"]], on="Sequence_ID", how="inner")
        mapping, ari, nmi, cm = align_labels_to_reference(merged, "ReferenceLabel", "psn_cluster")
        pd.Series(mapping).rename_axis("psn_cluster").to_frame("mapped_to_ref").to_csv(
            f"{out_prefix}_label_alignment.csv")
        cm.to_csv(f"{out_prefix}_confusion_matrix.csv")
        with open(f"{out_prefix}_agreement.txt", "w", encoding="utf-8") as f:
            f.write(f"Adjusted Rand Index (ARI): {ari:.4f}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi:.4f}\n")

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Clusters: {df_nodes['psn_cluster'].nunique()} ({method}, resolution={resolution})")
    print(f"Saved outputs with prefix '{out_prefix}'")


if __name__ == "__main__":
    main()
