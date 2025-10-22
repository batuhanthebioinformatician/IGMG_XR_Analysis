# filename: tm_align_matrix_docker.py
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, MMCIFParser, Select, PDBIO
from multiprocessing import Pool
from tqdm import tqdm

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
# Directory containing input PDB and/or CIF files.
workdir = r"/path/to/structures"
# Docker image providing TMalign.
docker_image = "biocontainers/tm-align:v20170708dfsg-2-deb_cv1"
# Number of worker processes for parallel alignment.
max_workers = 8
# Periodic and final TM-score checkpoint (square matrix CSV).
checkpoint_file = "tm_checkpoint.csv"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
class ChainASelect(Select):
    def accept_chain(self, chain):
        return 1 if chain.get_id() == "A" else 0


def extract_chainA(infile: str, outfile: str) -> bool:
    """Extract chain A from PDB/CIF; if absent, write the first chain."""
    ext = os.path.splitext(infile)[1].lower()
    parser = PDBParser(QUIET=True) if ext == ".pdb" else MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("struct", infile)
        io = PDBIO()
        chains = list(structure.get_chains())
        if any(c.get_id() == "A" for c in chains):
            io.set_structure(structure)
            io.save(outfile, select=ChainASelect())
        else:
            io.set_structure(chains[0])
            io.save(outfile)
        return True
    except Exception as e:
        print(f"Failed to parse {infile}: {e}")
        return False


def run_tmalign(args):
    """Run TMalign in Docker for one pair and return (f1, f2, tm_score)."""
    f1, f2, tmpdir = args
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{tmpdir}:/data",
        docker_image,
        "TMalign", f"/data/{f1}", f"/data/{f2}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tm_score = None
    for line in result.stdout.splitlines():
        if line.strip().startswith("TM-score="):
            try:
                tm_score = float(line.split()[1])
                break
            except Exception:
                continue
    return f1, f2, tm_score


def save_checkpoint(scores: dict, files: list[str]) -> None:
    """Write a symmetric TM-score matrix to CSV using collected pair scores."""
    n = len(files)
    mat = np.zeros((n, n))
    for i, f1 in enumerate(files):
        for j, f2 in enumerate(files):
            if i == j:
                mat[i, j] = 1.0
            elif (f1, f2) in scores:
                mat[i, j] = scores[(f1, f2)]
            elif (f2, f1) in scores:
                mat[i, j] = scores[(f2, f1)]
            else:
                mat[i, j] = np.nan
    df = pd.DataFrame(mat, index=files, columns=files)
    df.to_csv(checkpoint_file)


def load_checkpoint() -> pd.DataFrame | None:
    """Load an existing TM-score matrix checkpoint, if present."""
    if os.path.exists(checkpoint_file):
        return pd.read_csv(checkpoint_file, index_col=0)
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    tmpdir = tempfile.mkdtemp()

    # Extract chain A (or first chain) to temporary PDB files.
    chain_files = []
    for fname in os.listdir(workdir):
        if fname.lower().endswith((".pdb", ".cif")):
            infile = os.path.join(workdir, fname)
            outfile = os.path.join(tmpdir, os.path.splitext(fname)[0] + ".pdb")
            if extract_chainA(infile, outfile):
                chain_files.append(os.path.basename(outfile))
    print(f"Prepared {len(chain_files)} chain files.")

    # Load prior results if available.
    checkpoint = load_checkpoint()
    scores: dict[tuple[str, str], float] = {}
    if checkpoint is not None:
        print(f"Resuming from checkpoint: {checkpoint_file}")
        for i, f1 in enumerate(checkpoint.index):
            for j, f2 in enumerate(checkpoint.columns):
                if i < j:
                    val = checkpoint.iloc[i, j]
                    if pd.notna(val):
                        scores[(f1, f2)] = float(val)

    # Build remaining pair list.
    total_pairs = len(chain_files) * (len(chain_files) - 1) // 2
    pairs = []
    for i, f1 in enumerate(chain_files):
        for j in range(i + 1, len(chain_files)):
            f2 = chain_files[j]
            if (f1, f2) not in scores:
                pairs.append((f1, f2, tmpdir))

    print(f"Total pairs: {total_pairs} (remaining: {len(pairs)})")

    # Run TMalign in parallel.
    if pairs:
        with Pool(processes=max_workers) as pool:
            for f1, f2, tm in tqdm(pool.imap_unordered(run_tmalign, pairs), total=len(pairs)):
                scores[(f1, f2)] = tm
                if len(scores) % 50 == 0:
                    save_checkpoint(scores, chain_files)

    # Save final matrix.
    save_checkpoint(scores, chain_files)
    print(f"Done. Results saved to {checkpoint_file}")


if __name__ == "__main__":
    main()
