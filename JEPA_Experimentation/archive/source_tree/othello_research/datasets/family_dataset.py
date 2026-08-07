"""Reader + grouped sampler for the transposition-family artifacts (Part B).

Consumes ONLY the self-contained ``family_artifacts/`` directory produced by
``scripts/build_family_index.py``. No pickles or the original dataset are
needed at training time; prefixes are resolved through the packed token store
(O(1) random access via ``offsets.npy``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from othello_research.datasets.move_mapping import build_mappings

T_BANDS = ((4, 10), (11, 30), (31, 44))


def band_of(t: int) -> str:
    for lo, hi in T_BANDS:
        if lo <= t <= hi:
            return f"{lo}-{hi}"
    return "other"


def verify_manifest(artifacts_dir: Path) -> dict:
    """Verify every artifact's sha256 against manifest.json; raise on mismatch."""
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {artifacts_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for rel, expected in manifest.get("artifacts", {}).items():
        path = artifacts_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"Artifact listed in manifest missing: {rel}")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(
                f"Artifact sha256 mismatch for {rel}; the artifacts directory "
                "does not match its manifest. Rebuild or re-transfer it."
            )
    return manifest


@dataclass
class FamilyBatchConfig:
    board_size: int = 8
    families_per_batch: int = 16
    members_per_family: int = 16
    max_common_prefix_frac: float = 0.0  # 0 disables the leakage filter
    seed: int = 42


class FamilyIndexSplit:
    """In-memory view over one split (train/val) of the family index."""

    def __init__(self, artifacts_dir: Path, split: str) -> None:
        split_dir = artifacts_dir / "family_index" / split
        members = pq.read_table(split_dir / "members.parquet")
        self.family_id = members["family_id"].to_numpy()
        self.class_uid = members["class_uid"].to_numpy()
        self.game_id = members["game_id"].to_numpy()
        self.t = members["t"].to_numpy().astype(np.int64)
        self.common_prefix_len = members["common_prefix_len"].to_numpy()

        families = pq.read_table(split_dir / "families.parquet")
        fam_ids = families["family_id"].to_numpy()
        has_positive = families["has_positive"].to_numpy().astype(bool)
        n_classes = families["n_classes_kept"].to_numpy()
        # members.parquet is ordered by family_id; block boundaries:
        self.family_starts = np.searchsorted(self.family_id, fam_ids, side="left")
        self.family_ends = np.searchsorted(self.family_id, fam_ids, side="right")
        self.viable_families = fam_ids[(has_positive) & (n_classes >= 2)]
        self.fam_index_of = {int(f): i for i, f in enumerate(fam_ids)}

    def members_of(self, family_id: int) -> np.ndarray:
        i = self.fam_index_of[int(family_id)]
        return np.arange(self.family_starts[i], self.family_ends[i])


class FamilyArtifacts:
    """Packed token store + both splits, ready for sampling."""

    def __init__(
        self,
        artifacts_dir: str | Path,
        cfg: FamilyBatchConfig,
        *,
        verify: bool = True,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.cfg = cfg
        self.manifest = (
            verify_manifest(self.artifacts_dir)
            if verify
            else json.loads(
                (self.artifacts_dir / "manifest.json").read_text(encoding="utf-8")
            )
        )
        self.packed = np.memmap(
            self.artifacts_dir / "packed_games.bin", dtype=np.uint8, mode="r"
        )
        self.offsets = np.load(self.artifacts_dir / "offsets.npy")
        raw_to_token, _ = build_mappings(cfg.board_size)
        lut = np.full(cfg.board_size * cfg.board_size, -1, dtype=np.int64)
        for raw, token in enumerate(raw_to_token):
            lut[raw] = token
        self.raw_to_token = lut
        self.pad_token = cfg.board_size * cfg.board_size - 4
        self.train = FamilyIndexSplit(self.artifacts_dir, "train")
        self.val = FamilyIndexSplit(self.artifacts_dir, "val")

    def prefix_tokens(self, game_id: int, t: int) -> np.ndarray:
        start = self.offsets[game_id]
        raw = self.packed[start : start + t]
        tokens = self.raw_to_token[raw]
        if (tokens < 0).any():
            raise ValueError(
                f"Game {game_id} prefix contains an untokenizable square."
            )
        return tokens


class FamilyBatchSampler:
    """Infinite stream of G-families x <=S-members batches.

    Stratification within a family: two members of one positive-capable class
    (guaranteed anchors), at least one member of a different class (guaranteed
    negative), remaining slots uniform without replacement.
    """

    def __init__(
        self,
        artifacts: FamilyArtifacts,
        split: str,
        cfg: FamilyBatchConfig,
        seed_offset: int = 0,
    ) -> None:
        self.art = artifacts
        self.split: FamilyIndexSplit = getattr(artifacts, split)
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed + seed_offset)
        if len(self.split.viable_families) < cfg.families_per_batch:
            raise ValueError(
                f"Split has only {len(self.split.viable_families)} viable "
                f"families; need >= {cfg.families_per_batch} per batch."
            )
        self._queue: list[int] = []

    def _next_families(self) -> list[int]:
        while len(self._queue) < self.cfg.families_per_batch:
            order = self.split.viable_families.copy()
            self.rng.shuffle(order)
            self._queue.extend(int(f) for f in order)
        picked = self._queue[: self.cfg.families_per_batch]
        del self._queue[: self.cfg.families_per_batch]
        return picked

    def _sample_family_members(self, family_id: int) -> np.ndarray:
        split = self.split
        rows = split.members_of(family_id)
        classes: dict[int, list[int]] = {}
        for r in rows:
            classes.setdefault(int(split.class_uid[r]), []).append(int(r))
        S = self.cfg.members_per_family
        positive_classes = [c for c, m in classes.items() if len(m) >= 2]
        anchor_class = int(self.rng.choice(positive_classes))
        chosen: list[int] = list(
            self.rng.choice(classes[anchor_class], size=2, replace=False)
        )
        other_classes = [c for c in classes if c != anchor_class]
        other_class = int(self.rng.choice(other_classes))
        chosen.append(int(self.rng.choice(classes[other_class])))
        remaining = [r for r in rows if int(r) not in set(chosen)]
        n_fill = min(S - len(chosen), len(remaining))
        if n_fill > 0:
            chosen.extend(
                int(r)
                for r in self.rng.choice(remaining, size=n_fill, replace=False)
            )
        return np.asarray(chosen, dtype=np.int64)

    def sample_batch(self) -> dict[str, np.ndarray]:
        split = self.split
        rows: list[int] = []
        for family_id in self._next_families():
            rows.extend(self._sample_family_members(family_id))
        rows_arr = np.asarray(rows, dtype=np.int64)
        t = split.t[rows_arr]
        max_t = int(t.max())
        tokens = np.full((len(rows_arr), max_t), self.art.pad_token, dtype=np.int64)
        for i, r in enumerate(rows_arr):
            tokens[i, : t[i]] = self.art.prefix_tokens(
                int(split.game_id[r]), int(t[i])
            )
        cpl = split.common_prefix_len[rows_arr].astype(np.int64)
        frac = self.cfg.max_common_prefix_frac
        positive_eligible = (
            cpl <= (frac * t).astype(np.int64)
            if frac > 0
            else np.ones(len(rows_arr), dtype=bool)
        )
        return {
            "tokens": tokens,
            "lengths": t,
            "family_ids": split.family_id[rows_arr].astype(np.int64),
            "class_ids": split.class_uid[rows_arr].astype(np.int64),
            "positive_eligible": positive_eligible,
            "t_bands": np.asarray([band_of(int(x)) for x in t]),
        }
