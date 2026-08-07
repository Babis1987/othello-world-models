"""CLI trainer for transposition-family InfoNCE (Part B, GPU machine).

Consumes ONLY the ``family_artifacts/`` directory (via ``paths.artifacts_dir``)
produced by ``scripts/build_family_index.py`` on a different machine. Verifies
the manifest at startup and refuses to run on a mismatch.

Deliberately standalone: nothing here imports or modifies the train_jepa.py
chunk loop. Checkpoints are saved twice per milestone: a native ``latest.pt``
for resume, and an eval-compatible checkpoint that masquerades as a standard
``objective_class='jepa'`` (variant jepa_v1) model so the existing evaluation
notebooks load it unchanged (they only read ``context_encoder``).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from othello_research.datasets.family_dataset import (  # noqa: E402
    T_BANDS,
    FamilyArtifacts,
    FamilyBatchConfig,
    FamilyBatchSampler,
)
from othello_research.objectives.family_infonce import (  # noqa: E402
    FamilyInfoNCEConfig,
    OthelloFamilyInfoNCE,
)

KILL_SWITCH_README = """# Family-InfoNCE run notes

Kill-switch (check at step 5000):
  * held-out family retrieval top-1 must clearly exceed per-anchor chance, AND
  * excess = loss - plateau must be < 0 and still falling.
If either fails, ABORT the run. First rerun lever: set
`max_common_prefix_frac: 0.5` in the config (filters common-prefix positives).

Headline metric: `excess` in metrics.csv (≈0 ⇒ bag-of-moves collapse).
Note: excess < 0 is necessary but NOT sufficient evidence of color encoding;
confirm with the board-state probes in the eval notebook on a milestone
checkpoint before declaring success.
"""


@dataclass
class FamilyTrainConfig:
    artifacts_dir: str = ""
    out_dir: str = ""
    drive_sync_dir: str = ""
    verify_manifest: bool = True

    board_size: int = 8
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.1
    proj_hidden: int = 512
    proj_dim: int = 128

    temperature: float = 0.1
    families_per_batch: int = 16
    members_per_family: int = 16
    max_common_prefix_frac: float = 0.0

    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    total_steps: int = 20_000
    precision: str = "bf16"
    seed: int = 42

    log_every_steps: int = 100
    retrieval_every_steps: int = 500
    retrieval_anchors: int = 512
    milestone_steps: str = "5000,10000,15000,20000"
    drive_sync_every_steps: int = 1000

    resume: str = ""
    local_stage_dir: str = ""


def load_config(path: Path, overrides: dict[str, Any]) -> FamilyTrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    flat: dict[str, Any] = {}
    for section, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[section] = value
    known = {f.name for f in fields(FamilyTrainConfig)}
    kwargs = {k: v for k, v in flat.items() if k in known}
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return FamilyTrainConfig(**kwargs)


def run_with_shared_config(shared) -> None:
    """Entry point used by train_jepa.main() when objective_class=family_infonce.

    ``shared`` is a train_jepa.TrainConfig; only the fields relevant to the
    step-based family trainer are read. A smoke intent expressed through the
    chunk-loop vocabulary (``max_chunks``/tiny ``li_split_train_chunks``) is
    translated into a short step budget so the existing notebook smoke cells
    work unchanged.
    """
    total_steps = shared.total_steps if shared.total_steps > 0 else 20_000
    retrieval_every = shared.retrieval_every_steps
    retrieval_anchors = shared.retrieval_anchors
    if getattr(shared, "max_chunks", 0):
        total_steps = min(total_steps, 300)
        retrieval_every = min(retrieval_every, 100)
        retrieval_anchors = min(retrieval_anchors, 128)
        print(f"[family] smoke mode (max_chunks set): total_steps={total_steps}",
              flush=True)
    cfg = FamilyTrainConfig(
        artifacts_dir=shared.artifacts_dir,
        out_dir=shared.out_dir,
        drive_sync_dir=shared.drive_sync_dir or "",
        verify_manifest=shared.verify_manifest,
        board_size=shared.board_size,
        n_layers=shared.n_layers,
        n_heads=shared.n_heads,
        d_model=shared.d_model,
        dropout=shared.dropout,
        proj_hidden=shared.proj_hidden,
        proj_dim=shared.proj_dim,
        temperature=shared.temperature,
        families_per_batch=shared.families_per_batch,
        members_per_family=shared.members_per_family,
        max_common_prefix_frac=shared.max_common_prefix_frac,
        learning_rate=shared.learning_rate,
        min_lr_ratio=shared.min_lr_ratio,
        weight_decay=shared.weight_decay,
        beta1=shared.beta1,
        beta2=shared.beta2,
        grad_clip=shared.grad_clip,
        warmup_steps=shared.warmup_steps,
        total_steps=total_steps,
        precision=shared.precision,
        seed=shared.seed,
        log_every_steps=shared.log_every_steps,
        retrieval_every_steps=retrieval_every,
        retrieval_anchors=retrieval_anchors,
        milestone_steps=shared.milestone_steps,
        drive_sync_every_steps=shared.drive_sync_every_steps,
        resume=shared.resume or "",
        local_stage_dir=shared.local_stage_dir,
    )
    run(cfg)


def stage_artifacts_locally(cfg: FamilyTrainConfig) -> str:
    """Copy the artifacts dir to fast local disk when a stage dir is set."""
    if not cfg.local_stage_dir:
        return cfg.artifacts_dir
    src = Path(cfg.artifacts_dir)
    dst = Path(cfg.local_stage_dir)
    if src.resolve() == dst.resolve():
        return cfg.artifacts_dir
    print(f"[stage] copying artifacts {src} -> {dst}", flush=True)
    copied = 0
    for path in sorted(src.rglob("*")):
        if path.is_dir() or "_intermediate" in path.parts:
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copy2(path, target)
            copied += 1
    print(f"[stage] {copied} file(s) copied", flush=True)
    return str(dst)


def cosine_lr(step: int, cfg: FamilyTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(
        1, cfg.total_steps - cfg.warmup_steps
    )
    progress = min(1.0, progress)
    floor = cfg.learning_rate * cfg.min_lr_ratio
    return floor + 0.5 * (cfg.learning_rate - floor) * (
        1 + math.cos(math.pi * progress)
    )


def sync_dir(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for item in src.iterdir():
        if not item.is_file():
            continue
        target = dst / item.name
        if not target.exists() or item.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(item, target)
            copied += 1
    return copied


def save_eval_compat_checkpoint(
    model: OthelloFamilyInfoNCE,
    path: Path,
    step: int,
    rows_seen: int,
) -> None:
    """Save weights loadable by the EXISTING eval notebooks.

    build_model_from_checkpoint dispatches on model_config.objective_class,
    whose registry we may not modify (additive-only requirement). We therefore
    instantiate a standard shared-encoder OthelloJEPA (jepa_v1, linear
    predictor), copy the trained encoder into its context_encoder, and save
    that module's state_dict. The eval paths only ever read context_encoder;
    the untrained linear predictor weights are inert. The projection head is
    stored under a separate key that no existing loader touches.
    """
    from othello_research.objectives.jepa import JEPAConfig, OthelloJEPA

    jepa_cfg = JEPAConfig(
        variant="jepa_v1",
        loss_type="vicreg",
        view_mode="nested",
        board_size=model.cfg.board_size,
        n_layers=model.cfg.n_layers,
        n_heads=model.cfg.n_heads,
        d_model=model.cfg.d_model,
        dropout=model.cfg.dropout,
        predictor_type="linear",
        prediction_horizon=1,
        use_ema_target=False,
    )
    shell = OthelloJEPA(jepa_cfg)
    shell.context_encoder.load_state_dict(model.context_encoder.state_dict())
    torch.save(
        {
            "model": shell.state_dict(),
            "model_config": {"objective_class": "jepa", **asdict(jepa_cfg)},
            "jepa_config": asdict(jepa_cfg),
            "train_config": {
                "objective_class": "jepa",
                "variant": "jepa_v1",
                "family_infonce_source": True,
            },
            "family_infonce_config": asdict(model.cfg),
            "projection_head": model.projection.state_dict(),
            "step": step,
            "games_seen": rows_seen,
        },
        path,
    )


@torch.no_grad()
def retrieval_eval(
    model: OthelloFamilyInfoNCE,
    sampler: FamilyBatchSampler,
    device: torch.device,
    n_anchors: int,
    autocast_ctx,
) -> dict[str, float]:
    """Held-out top-1 family retrieval vs per-anchor chance, per t-band."""
    model.eval()
    hits: dict[str, list[float]] = {b: [] for b in
                                    [f"{lo}-{hi}" for lo, hi in T_BANDS]}
    chances: dict[str, list[float]] = {b: [] for b in hits}
    collected = 0
    while collected < n_anchors:
        batch = sampler.sample_batch()
        tokens = torch.from_numpy(batch["tokens"]).to(device)
        lengths = torch.from_numpy(batch["lengths"]).to(device)
        family_ids = torch.from_numpy(batch["family_ids"]).to(device)
        class_ids = torch.from_numpy(batch["class_ids"]).to(device)
        with autocast_ctx():
            z = model.project(model.encode_states(tokens, lengths)).float()
        sim = z @ z.t()
        eye = torch.eye(sim.size(0), dtype=torch.bool, device=device)
        cand = (family_ids.unsqueeze(0) == family_ids.unsqueeze(1)) & ~eye
        same_class = (
            class_ids.unsqueeze(0) == class_ids.unsqueeze(1)
        ) & cand
        n_pos = same_class.sum(1)
        n_neg = (cand & ~same_class).sum(1)
        anchors = torch.nonzero((n_pos >= 1) & (n_neg >= 1)).squeeze(-1)
        masked = sim.masked_fill(~cand, torch.finfo(sim.dtype).min)
        top1 = masked.argmax(dim=1)
        bands = batch["t_bands"]
        for a in anchors.tolist():
            band = str(bands[a])
            if band not in hits:
                continue
            hits[band].append(float(same_class[a, top1[a]]))
            chances[band].append(
                float(n_pos[a]) / float(cand[a].sum())
            )
            collected += 1
        if collected >= n_anchors:
            break
    model.train()
    result: dict[str, float] = {}
    all_hits: list[float] = []
    all_chance: list[float] = []
    for band in hits:
        if hits[band]:
            result[f"top1_{band}"] = float(np.mean(hits[band]))
            result[f"chance_{band}"] = float(np.mean(chances[band]))
            all_hits.extend(hits[band])
            all_chance.extend(chances[band])
    result["top1_overall"] = float(np.mean(all_hits)) if all_hits else float("nan")
    result["chance_overall"] = (
        float(np.mean(all_chance)) if all_chance else float("nan")
    )
    result["n_anchors"] = float(len(all_hits))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifacts_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--drive_sync_dir", type=str, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--retrieval_every_steps", type=int, default=None)
    parser.add_argument("--no_verify_manifest", action="store_true")
    args = parser.parse_args(argv)

    overrides = {
        "artifacts_dir": args.artifacts_dir,
        "out_dir": args.out_dir,
        "drive_sync_dir": args.drive_sync_dir,
        "total_steps": args.total_steps,
        "resume": args.resume,
        "seed": args.seed,
        "retrieval_every_steps": args.retrieval_every_steps,
    }
    cfg = load_config(args.config, overrides)
    if args.no_verify_manifest:
        cfg.verify_manifest = False
    run(cfg)


def run(cfg: FamilyTrainConfig) -> None:
    if not cfg.artifacts_dir:
        raise SystemExit("paths.artifacts_dir is required")
    if not cfg.out_dir:
        raise SystemExit("paths.out_dir is required")
    cfg.artifacts_dir = stage_artifacts_locally(cfg)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda" and cfg.precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    def autocast_ctx():
        return torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(KILL_SWITCH_README, encoding="utf-8")

    batch_cfg = FamilyBatchConfig(
        board_size=cfg.board_size,
        families_per_batch=cfg.families_per_batch,
        members_per_family=cfg.members_per_family,
        max_common_prefix_frac=cfg.max_common_prefix_frac,
        seed=cfg.seed,
    )
    print(f"[artifacts] loading {cfg.artifacts_dir} "
          f"(verify_manifest={cfg.verify_manifest})", flush=True)
    artifacts = FamilyArtifacts(
        cfg.artifacts_dir, batch_cfg, verify=cfg.verify_manifest
    )
    train_sampler = FamilyBatchSampler(artifacts, "train", batch_cfg)
    val_sampler = FamilyBatchSampler(artifacts, "val", batch_cfg,
                                     seed_offset=1_000_003)

    model_cfg = FamilyInfoNCEConfig(
        board_size=cfg.board_size,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        dropout=cfg.dropout,
        proj_hidden=cfg.proj_hidden,
        proj_dim=cfg.proj_dim,
        temperature=cfg.temperature,
    )
    model = OthelloFamilyInfoNCE(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    step = 0
    rows_seen = 0
    resume_path = Path(cfg.resume) if cfg.resume else out_dir / "latest.pt"
    if resume_path.is_file():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))
        rows_seen = int(ckpt.get("rows_seen", 0))
        print(f"[resume] from {resume_path} at step={step}", flush=True)

    milestones = {
        int(x) for x in str(cfg.milestone_steps).split(",") if str(x).strip()
    }
    metrics_path = out_dir / "metrics.csv"
    retrieval_path = out_dir / "retrieval.csv"
    write_header = not metrics_path.exists()
    metrics_file = open(metrics_path, "a", newline="", encoding="utf-8")
    metrics_writer = csv.writer(metrics_file)
    if write_header:
        metrics_writer.writerow(
            ["time", "step", "loss", "plateau", "excess", "n_anchors",
             "emb_std", "lr", "rows_seen"]
        )
    retrieval_header = not retrieval_path.exists()

    def save_native(path: Path, include_optimizer: bool = True) -> None:
        # model_config makes these checkpoints loadable through the existing
        # build_model_from_checkpoint registry, so every eval notebook works
        # on them by changing only its CONFIG_NAME.
        payload = {
            "model": model.state_dict(),
            "model_config": {
                "objective_class": "family_infonce",
                **asdict(model_cfg),
            },
            "train_config": {
                "objective_class": "family_infonce",
                "variant": "family_infonce",
            },
            "family_train_config": asdict(cfg),
            "family_infonce_config": asdict(model_cfg),
            "step": step,
            "games_seen": rows_seen,
            "rows_seen": rows_seen,
        }
        if include_optimizer:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    print(f"[train] steps={cfg.total_steps} batch="
          f"{cfg.families_per_batch}x{cfg.members_per_family} device={device}",
          flush=True)
    t_start = time.monotonic()
    window_loss: list[float] = []
    window_plateau: list[float] = []
    window_std: list[float] = []

    model.train()
    while step < cfg.total_steps:
        batch = train_sampler.sample_batch()
        tokens = torch.from_numpy(batch["tokens"]).to(device)
        lengths = torch.from_numpy(batch["lengths"]).to(device)
        family_ids = torch.from_numpy(batch["family_ids"]).to(device)
        class_ids = torch.from_numpy(batch["class_ids"]).to(device)
        eligible = torch.from_numpy(batch["positive_eligible"]).to(device)

        lr = cosine_lr(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            out = model(tokens, lengths, family_ids, class_ids, eligible)
            loss = out["loss"]
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        rows_seen += tokens.size(0)
        step += 1
        window_loss.append(float(loss.detach()))
        window_plateau.append(float(out["plateau"]))
        window_std.append(float(out["emb_std"]))

        if step % cfg.log_every_steps == 0:
            mean_loss = float(np.mean(window_loss))
            mean_plateau = float(np.mean(window_plateau))
            elapsed = time.monotonic() - t_start
            steps_per_sec = step / max(1e-9, elapsed)
            eta_min = (cfg.total_steps - step) / max(1e-9, steps_per_sec) / 60
            metrics_writer.writerow(
                [time.strftime("%Y-%m-%d %H:%M:%S"), step,
                 f"{mean_loss:.6f}", f"{mean_plateau:.6f}",
                 f"{mean_loss - mean_plateau:.6f}",
                 int(out["n_anchors"]), f"{float(np.mean(window_std)):.4f}",
                 f"{lr:.3e}", rows_seen]
            )
            metrics_file.flush()
            print(
                f"[step {step:6d}/{cfg.total_steps}] loss={mean_loss:.4f} "
                f"plateau={mean_plateau:.4f} "
                f"excess={mean_loss - mean_plateau:+.4f} "
                f"{steps_per_sec:.2f} it/s ETA={eta_min:.0f}m",
                flush=True,
            )
            window_loss.clear()
            window_plateau.clear()
            window_std.clear()

        if step % cfg.retrieval_every_steps == 0:
            result = retrieval_eval(
                model, val_sampler, device, cfg.retrieval_anchors, autocast_ctx
            )
            keys = sorted(result)
            with open(retrieval_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if retrieval_header:
                    writer.writerow(["step", *keys])
                    retrieval_header = False
                writer.writerow([step, *[f"{result[k]:.4f}" for k in keys]])
            print(
                f"[retrieval @ {step}] top1={result['top1_overall']:.4f} "
                f"chance={result['chance_overall']:.4f}",
                flush=True,
            )

        if step in milestones or step == cfg.total_steps:
            save_native(out_dir / "latest.pt")
            save_native(out_dir / f"milestone_step{step}.pt")
            save_eval_compat_checkpoint(
                model, out_dir / f"eval_compat_step{step}.pt", step, rows_seen
            )
            save_eval_compat_checkpoint(
                model, out_dir / "eval_compat_latest.pt", step, rows_seen
            )
            print(f"[checkpoint] step={step}", flush=True)
        elif step % 1000 == 0:
            save_native(out_dir / "latest.pt")

        if cfg.drive_sync_dir and step % cfg.drive_sync_every_steps == 0:
            copied = sync_dir(out_dir, Path(cfg.drive_sync_dir))
            if copied:
                print(f"[drive sync] {copied} file(s)", flush=True)

    metrics_file.close()
    save_native(out_dir / "final.pt", include_optimizer=False)
    save_eval_compat_checkpoint(model, out_dir / "final_eval_compat.pt",
                                step, rows_seen)
    if cfg.drive_sync_dir:
        sync_dir(out_dir, Path(cfg.drive_sync_dir))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
