# filename: shannon_entropy_no_gaps.py
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


# ------------------------------------------------------------------
# Load MSA from FASTA
# ------------------------------------------------------------------
def read_fasta_alignment(filepath: str) -> list[str]:
    """Return a list of sequences from a FASTA alignment file."""
    sequences = []
    current_seq = []
    with open(filepath, "r") as handle:
        for line in handle:
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.strip())
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


# ------------------------------------------------------------------
# Shannon entropy (natural log), ignoring gaps
# ------------------------------------------------------------------
def shannon_entropy_ln_no_gaps(column: np.ndarray) -> float:
    """Compute Shannon entropy using natural log for one alignment column, ignoring '-'."""
    counts = defaultdict(int)
    for symbol in column:
        if symbol != "-":
            counts[symbol] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    return -sum((cnt / total) * math.log(cnt / total) for cnt in counts.values())


# ------------------------------------------------------------------
# Compute entropy for each alignment position
# ------------------------------------------------------------------
fasta_path = r"/path/to/cluster_alignment.fasta"  # update with your file path
seqs = read_fasta_alignment(fasta_path)

alignment = np.array([list(s) for s in seqs])
n_positions = alignment.shape[1]

entropy_vals = [shannon_entropy_ln_no_gaps(alignment[:, i]) for i in range(n_positions)]

# ------------------------------------------------------------------
# Save results to Excel
# ------------------------------------------------------------------
# Requires openpyxl or xlsxwriter installed.
out_path = "cluster_shannon_entropy_no_gaps.xlsx"

df = pd.DataFrame(
    {
        "Position": np.arange(1, n_positions + 1),
        "Shannon_Entropy_ln_no_gaps": entropy_vals,
    }
)

df.to_excel(out_path, index=False, sheet_name="Entropy")
print(f"Entropy table saved → {out_path}")

# ------------------------------------------------------------------
# Plot entropy profile
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(range(1, n_positions + 1), entropy_vals)
plt.xlabel("Alignment Position")
plt.ylabel("Shannon Entropy (ln, gaps ignored)")
plt.title("Shannon Entropy per Position")
plt.grid(True)
plt.tight_layout()
plt.show()
