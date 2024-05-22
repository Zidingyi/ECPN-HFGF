import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import numpy as np


class ECPNDataset(Dataset):
    def __init__(self, datapath_fasta: str):
        self.accessions, self.sequences = [], []
        with open(datapath_fasta, 'r') as f:
            for seq_record in SeqIO.parse(f, 'fasta'):
                self.sequences.append(str(seq_record.seq))
                self.accessions.append(seq_record.id)
        self.kv = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                   'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20}

    def __len__(self):
        return len(self.accessions)

    def __getitem__(self, idx: int):
        sequence = self.sequences[idx]
        # replace bad letter
        sequence = sequence.replace('U', 'X')
        sequence = sequence.replace('O', 'X')
        if 'B' in sequence:  # 替换未被编码的B和Z
            if np.random.random() > 0.5:
                sequence = sequence.replace('B', 'D')
            else:
                sequence = sequence.replace('B', 'N')
        if 'Z' in sequence:
            if np.random.random() > 0.5:
                sequence = sequence.replace('Z', 'E')
            else:
                sequence = sequence.replace('Z', 'Q')

        x = torch.zeros(1000, 21)
        for i, s in enumerate(sequence):
            x[i][self.kv[s]] = 1.

        x = x.reshape((1,) + x.shape)
        return self.accessions[idx], x
