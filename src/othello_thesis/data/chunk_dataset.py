"""In-memory dataset for one chunk of Othello games.

This module follows the chunked-training pattern used in the Go-GPT
project: the training loop loads one shard at a time, wraps it in a fresh
dataset, trains for one pass, and frees memory before moving to the next
shard. Random access is safe and fast because the entire chunk lives in
RAM during its own pass.

Vocab layout (for an nÃ—n board):
    0 .. nÂ²-5      â€” valid move tokens (nÂ²-4 values)
    nÂ²-4           â€” input padding token
    Total vocab_size = nÂ²-3

Two distinct pad mechanisms are used (one for inputs, one for targets):
    - Input padding token: a real embedding index, sits just past the
      move vocabulary.
    - Target padding sentinel TARGET_PAD = -100: not a real token; the
      default behaviour of torch.nn.functional.cross_entropy is to skip
      positions equal to -100, so they contribute nothing to the loss.

Typical use in the training loop:

    for chunk_path in shuffled(chunk_paths):
        games = load_chunk(chunk_path)
        dataset = OthelloChunkDataset(games, block_size, board_size)
        loader = DataLoader(dataset, batch_size=256, shuffle=True,
                            num_workers=2, pin_memory=True)
        train_on_loader(model, loader, ...)
        del games, dataset, loader
"""

from __future__ import annotations

import pickle

import torch
from torch.utils.data import Dataset

from othello_thesis.data.move_vocabulary import build_mappings


# Sentinel value used in the target tensor for padded positions.
# Cross-entropy loss with default ignore_index=-100 will skip these.
TARGET_PAD = -100


def load_chunk(path: str) -> list[list[int]]:
    """Load one pickle shard into memory.

    Args:
        path: filesystem path to a .pickle file containing a list of
            games, where each game is a list of raw board-position ints.

    Returns:
        The deserialised list-of-lists.
    """
    with open(path, "rb") as f:
        return pickle.load(f)


class OthelloChunkDataset(Dataset):
    """In-memory dataset wrapping a single chunk of pre-loaded games.

    Designed to be created, consumed by one training pass, and discarded.
    Random access is O(1) since all games live in RAM.

    Each item returns (x, y) for next-token prediction:
        x = tokens[0:block_size]      (input sequence)
        y = tokens[1:block_size+1]    (target sequence, shifted by 1)

    Both x and y are remapped from raw positions to contiguous token ids
    (0..nÂ²-5). Padding handling:
        - Short games are padded so all sequences have length block_size+1.
        - In x, padded positions hold input_pad_token (a valid embedding
          index just past the move vocabulary).
        - In y, padded positions hold TARGET_PAD = -100, ignored by
          cross_entropy.
    """

    def __init__(
        self,
        games: list[list[int]],
        block_size: int,
        board_size: int,
    ):
        self.games = games
        self.block_size = block_size
        self.board_size = board_size

        # Token id used to pad model inputs. Sits immediately after the
        # nÂ²-4 valid move tokens (0..nÂ²-5), at index nÂ²-4.
        self.input_pad_token = board_size * board_size - 4

        # Pre-compute the rawâ†’token mapping for this board size.
        self._raw_to_token, _ = build_mappings(board_size)

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= idx < len(self.games)):
            raise IndexError(f"Index {idx} out of range [0, {len(self.games)})")

        raw_seq = self.games[idx]

        # Remap raw positions (0..nÂ²-1) to token ids (0..nÂ²-5).
        # Starting positions (which have no token) are a data-integrity
        # error and would later crash the embedding lookup with a cryptic
        # device-side assert. We surface them here with a clear message.
        seq: list[int] = []
        for m in raw_seq:
            if not (0 <= m < self.board_size * self.board_size):
                raise ValueError(
                    f"Game at index {idx} contains out-of-range move "
                    f"({m}); expected a raw board position in "
                    f"[0, {self.board_size * self.board_size})."
                )

            t = self._raw_to_token[m]
            if t == -1:
                raise ValueError(
                    f"Game at index {idx} contains a starting position "
                    f"({m}), which is not a legal Othello move."
                )
            seq.append(t)

        # Truncate to block_size + 1 (block_size for x, +1 for shifted y).
        seq = seq[: self.block_size + 1]

        # Pad short games with TARGET_PAD; we'll fix the input side below.
        pad_len = (self.block_size + 1) - len(seq)
        if pad_len > 0:
            seq = seq + [TARGET_PAD] * pad_len

        seq_t = torch.tensor(seq, dtype=torch.long)
        x, y = seq_t[:-1], seq_t[1:]

        # In the input tensor, replace TARGET_PAD with a valid embedding
        # index. The target tensor keeps TARGET_PAD so cross_entropy
        # ignores those positions in the loss.
        x = x.masked_fill(x == TARGET_PAD, self.input_pad_token)

        return x, y
