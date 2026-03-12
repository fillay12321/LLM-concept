from __future__ import annotations

from typing import List, Tuple

import torch

from .tokenizer import TiktokenWrapper


class BookDataset:
    def __init__(
        self,
        file_paths: List[str],
        tokenizer: TiktokenWrapper,
        seq_len: int = 256,
        stride: int = 128,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if stride > seq_len:
            raise ValueError("stride must be <= seq_len")

        self.file_paths = list(file_paths)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.stride = int(stride)

        flat: List[int] = []
        for p in self.file_paths:
            flat.extend(self.tokenizer.encode_file(p))

        self.all_tokens = torch.tensor(flat, dtype=torch.long)
        self.seqs = self._make_sequences(self.all_tokens, self.seq_len, self.stride)

    @staticmethod
    def _make_sequences(all_tokens: torch.Tensor, seq_len: int, stride: int) -> torch.Tensor:
        if all_tokens.numel() < seq_len:
            return torch.empty((0, seq_len), dtype=torch.long)

        starts = list(range(0, int(all_tokens.numel() - seq_len + 1), int(stride)))
        out = torch.empty((len(starts), seq_len), dtype=torch.long)
        for i, s in enumerate(starts):
            out[i] = all_tokens[s : s + seq_len]
        return out

    @classmethod
    def _from_seqs(
        cls,
        seqs: torch.Tensor,
        tokenizer: TiktokenWrapper,
        seq_len: int,
        stride: int,
    ) -> "BookDataset":
        obj = cls.__new__(cls)
        obj.file_paths = []
        obj.tokenizer = tokenizer
        obj.seq_len = int(seq_len)
        obj.stride = int(stride)
        obj.all_tokens = torch.empty((0,), dtype=torch.long)
        obj.seqs = seqs.to(dtype=torch.long)
        return obj

    def __len__(self) -> int:
        return int(self.seqs.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.seqs[idx]

    def train_val_split(self, val_ratio: float = 0.1) -> Tuple["BookDataset", "BookDataset"]:
        if not (0.0 < val_ratio < 1.0):
            raise ValueError("val_ratio must be in (0, 1)")

        n = len(self)
        if n == 0:
            return (
                BookDataset._from_seqs(self.seqs.clone(), self.tokenizer, self.seq_len, self.stride),
                BookDataset._from_seqs(self.seqs.clone(), self.tokenizer, self.seq_len, self.stride),
            )

        n_val = max(1, int(round(n * val_ratio)))
        n_train = max(0, n - n_val)

        train_seqs = self.seqs[:n_train]
        val_seqs = self.seqs[n_train:]

        return (
            BookDataset._from_seqs(train_seqs, self.tokenizer, self.seq_len, self.stride),
            BookDataset._from_seqs(val_seqs, self.tokenizer, self.seq_len, self.stride),
        )

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)


def prepare_books(
    file_paths: List[str],
    seq_len: int = 256,
    stride: int = 128,
) -> Tuple[BookDataset, BookDataset]:
    tok = TiktokenWrapper()
    ds = BookDataset(file_paths=file_paths, tokenizer=tok, seq_len=seq_len, stride=stride)
    train_ds, val_ds = ds.train_val_split(val_ratio=0.1)
    return train_ds, val_ds
