# filename: XR_blosum45_matrix.py
import multiprocessing as mp
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import AlignIO
from Bio.Align import substitution_matrices

# ---------------------------------------------------------------------------
# User-configurable settings
# ---------------------------------------------------------------------------
# Path to the multiple sequence alignment (MSA) file.
msa_path = r"/path/to/alignment.fasta"
# Alignment format (e.g., "fasta", "clustal").
msa_format = "fasta"
# Score to use when at least one position is a gap.
gap_penalty = -5
# Output file prefix (without extension).
out_prefix = r"/path/to/output/XR_BLOSUM45"
# Number of worker processes (None → CPU count minus one).
n_workers = None
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_blosum45_lookup(gap_score: int) -> np.ndarray:
    """Build a 91×91 ASCII-indexed lookup table containing BLOSUM45 scores.
    Any pair not present in the matrix (e.g., gaps) is assigned gap_score.
    """
    mat = np.full((91, 91), gap_score, dtype=np.int16)
    bl45 = substitution_matrices.load("BLOSUM45")
    for (a1, a2), score in bl45.items():
        mat[ord(a1), ord(a2)] = score
        mat[ord(a2), ord(a1)] = score
    return mat


def score_pair(pair, aln_ascii: np.ndarray, blosum: np.ndarray):
    """Return (i, j, total_blosum_score) for a sequence pair (i, j)."""
    i, j = pair
    return i, j, int(blosum[aln_ascii[i], aln_ascii[j]].sum())


def score_pair_wrapper(args):
    """Wrapper for multiprocessing pickling on Windows."""
    pair, aln_ascii, blosum = args
    return score_pair(pair, aln_ascii, blosum)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def build_similarity_and_distance(
    msa_file: str,
    fmt: str,
    gap_score: int,
    workers: int | None = None,
):
    """Compute pairwise BLOSUM45 similarity and a derived distance matrix.

    Distance is defined as (max off-diagonal similarity) − (pairwise similarity).
    Diagonal distances are set to 0.
    Returns:
        ids: list of sequence IDs
        sim: (n×n) int32 similarity matrix
        dist: (n×n) int32 distance matrix
    """
    aln = AlignIO.read(msa_file, fmt)
    n_seq = len(aln)
    aln_len = aln.get_alignment_length()
    ids = [rec.id for rec in aln]

    # Alignment to ASCII codes: shape (n_seq, aln_len)
    aln_ascii = np.frombuffer(
        "".join(str(rec.seq) for rec in aln).encode("ascii"), dtype=np.uint8
    ).reshape(n_seq, aln_len)

    blosum = build_blosum45_lookup(gap_score)
    pairs = list(combinations(range(n_seq), 2))

    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    sim = np.zeros((n_seq, n_seq), dtype=np.int32)

    # Parallel pair scoring
    args = [(p, aln_ascii, blosum) for p in pairs]
    with mp.Pool(processes=workers) as pool:
        for i, j, score in tqdm(
            pool.imap_unordered(score_pair_wrapper, args),
            total=len(pairs),
            desc=f"Scoring {len(pairs):,} pairs on {workers} cores",
        ):
            sim[i, j] = sim[j, i] = score

    # Self-similarity (diagonal)
    for i in range(n_seq):
        sim[i, i] = blosum[aln_ascii[i], aln_ascii[i]].sum()

    # Max off-diagonal similarity
    max_sim = np.max(sim[~np.eye(n_seq, dtype=bool)])

    # Distance definition
    dist = (max_sim - sim).astype(np.int32)
    np.fill_diagonal(dist, 0)

    return ids, sim, dist


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ids, sim_mat, dist_mat = build_similarity_and_distance(
        msa_file=msa_path,
        fmt=msa_format,
        gap_score=gap_penalty,
        workers=n_workers,
    )

    df_sim = pd.DataFrame(sim_mat, index=ids, columns=ids)
    df_dist = pd.DataFrame(dist_mat, index=ids, columns=ids)

    sim_csv = Path(f"{out_prefix}_similarity.csv").resolve()
    dist_csv = Path(f"{out_prefix}_distance.csv").resolve()

    df_sim.to_csv(sim_csv)
    df_dist.to_csv(dist_csv)

    print("Similarity matrix saved to:", sim_csv)
    print("Distance matrix saved to:", dist_csv)


if __name__ == "__main__":
    main()
