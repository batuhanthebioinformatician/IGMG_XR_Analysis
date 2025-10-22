Comparative Genomics and Structural Analysis of Phototrophicity Across Diverse Environments — Machine Learning, Entropy, and Graph-Based Applications

This repository provides computational tools integrating machine learning, information theory, and graph theory for comparative genomic and structural analysis of phototrophic proteins, with emphasis on xanthorhodopsins (XRs).
The scripts implement quantitative similarity scoring, unsupervised clustering, entropy profiling, and graph-based community detection to study functional and evolutionary patterns across diverse environments.

Repository Overview
Script	Description
xr_blosum45_matrix.py	Computes a pairwise BLOSUM45 similarity and distance matrix from a multiple sequence alignment (MSA). Serves as a foundation for clustering and graph construction.
xr_hdbscan_umap.py	Applies unsupervised clustering (HDBSCAN) to the similarity matrix and visualizes the relationships among proteins in 2D and 3D UMAP embeddings.
tm_align_matrix_docker.py	Performs all-versus-all structural alignment using TM-align inside a Docker container to generate a symmetric TM-score matrix for structural comparison.
shannon_entropy_no_gaps.py	Calculates position-wise Shannon entropy (natural log, gaps ignored) from an MSA to measure residue-level information content and evolutionary variability.
build_psn_and_cluster.py	Builds a Protein Similarity Network (PSN) from a BLOSUM45 similarity matrix. Supports k-nearest-neighbor and threshold-based graph construction, followed by community detection (Leiden or Louvain). Outputs graph files for Gephi or Cytoscape visualization.

Workflow Summary

1.Sequence Similarity and Information Content

.Compute similarity and distance matrices using xr_blosum45_matrix.py.

.Analyze sequence entropy using shannon_entropy_no_gaps.py.

2.Unsupervised Clustering and Dimensionality Reduction

.Perform HDBSCAN-based clustering and UMAP visualization using xr_hdbscan_umap.py.

3.Structural Comparison

.Use tm_align_matrix_docker.py to quantify 3D structural similarity among modeled or experimentally solved proteins.

4.Network and Graph Analysis

.Construct protein similarity networks with build_psn_and_cluster.py.

.Detect community structures reflecting potential functional or evolutionary groupings.

Key Features

BLOSUM-based sequence similarity computation

Entropy-based analysis of residue variability

Unsupervised clustering (HDBSCAN, UMAP)

TM-align–based structural similarity assessment

Graph-theoretic representation and community detection (Leiden or Louvain)

Compatible with large protein datasets and multi-environment comparative studies

Requirements

Python ≥ 3.9

Required libraries: numpy, pandas, biopython, tqdm, networkx, matplotlib, umap-learn, hdbscan, community, leidenalg, igraph, scikit-learn

Docker (for TM-align structural comparisons)

Installation and Repository Access
Clone the Repository

To obtain a local copy of this repository:

git clone https://github.com/batuhanthebioinformatician/IGMG_XR_Analysis.git
cd IGMG_XR_Analysis
