from pathlib import Path
import subprocess
import shutil
import numpy as np
import torch


CHAR_TO_INDEX = {
    "A": 0,
    "C": 1,
    "T": 2,
    "G": 3,
    "(": 4,
    ".": 5,
    ")": 6,
}


def load_data(sequences_input,labels_input):
    """
    Load sequences and regression labels.

    Expected files:
        dataset_dir/
            cdGII_seqs.txt
            cdGII_labels.txt
    """
    # dataset_dir = Path(dataset_dir)

    seqs = Path(sequences_input).read_text().splitlines()
    labels = np.loadtxt(Path(labels_input))

    return seqs, labels


def pad_sequences(sequences, length=113, pad_char="A"):
    """
    Pad or truncate sequences to a fixed length.
    """
    return [
        seq[:length] + pad_char * max(0, length - len(seq))
        for seq in sequences
    ]

import sys
from pathlib import Path

def find_rnafold():
    """
    Locate RNAfold from the current Python environment.
    Works for conda environments and system installations.
    """

    # 1. Check PATH first
    rnafold = shutil.which("RNAfold")
    if rnafold is not None:
        return rnafold

    # 2. Infer conda environment from Python executable
    python_path = Path(sys.executable)
    conda_bin = python_path.parent
    candidate = conda_bin / "RNAfold"

    if candidate.exists():
        return str(candidate)

    raise RuntimeError(
        "RNAfold not found. Please install ViennaRNA "
        "(e.g. conda install -c bioconda viennarna)."
    )

def run_rnafold(sequences, fasta_file="generated.fasta"):

    rnafold = find_rnafold()

    with open(fasta_file, "w") as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")

    result = subprocess.run(
        [rnafold, "--noPS", fasta_file],
        capture_output=True,
        text=True,
        check=True
    )

    structures = []

    for line in result.stdout.splitlines():
        if "(" in line:
            structures.append(line.split()[0])

    return structures


def encode_sequences(sequences, structures):
    """
    Encode sequence and secondary structure into one-hot representation.

    Output shape:
        (N, sequence_length, 7)
    """

    n = len(sequences)
    L = len(sequences[0])

    encoded = np.zeros((n, L, len(CHAR_TO_INDEX)), dtype=np.float32)

    for i, (seq, struct) in enumerate(zip(sequences, structures)):

        if len(seq) != len(struct):
            raise ValueError(
                f"Sequence {i} and structure have different lengths."
            )

        for j, (nt, ss) in enumerate(zip(seq, struct)):

            if nt not in CHAR_TO_INDEX:
                raise ValueError(f"Unknown nucleotide: {nt}")

            if ss not in CHAR_TO_INDEX:
                raise ValueError(f"Unknown structure symbol: {ss}")

            encoded[i, j, CHAR_TO_INDEX[nt]] = 1
            encoded[i, j, CHAR_TO_INDEX[ss]] = 1

    return encoded


def preprocess_dataset(
    sequences_input="./datasets/cdGII_seqs.txt",
    labels_input="./datasets/cdGII_labels.txt",
    sequence_length=113,
):
    """
    Complete preprocessing pipeline.

    Returns
    -------
    X : torch.FloatTensor
        Shape (N, 7, sequence_length)

    y : torch.FloatTensor
        Shape (N,)
    """

    sequences, labels = load_data(sequences_input,labels_input)

    sequences = pad_sequences(
        sequences,
        length=sequence_length,
    )

    structures = run_rnafold(sequences)

    X = encode_sequences(
        sequences,
        structures,
    )

    X = torch.tensor(
        np.transpose(X, (0, 2, 1)),
        dtype=torch.float32,
    )

    y = torch.tensor(
        labels,
        dtype=torch.float32,
    )

    return X, y