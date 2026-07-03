"""Train teacher networks from guided trajectories (Stage B).

Each teacher is a small network that imitates one guidance source. It is trained
by supervised cross-entropy on the (raw local_obs, agent_index) -> guide action
pairs produced by generate_guided_trajectories.py.

Typical use (after generating both datasets):

    python3 train_teachers.py --both

which trains:
    distill_data/real         -> teachers/teacher_llm.pt
    distill_data/best_preview -> teachers/teacher_heur.pt

Or train a single teacher:

    python3 train_teachers.py --dataset-dir distill_data/best_preview \
        --out-path teachers/teacher_heur.pt --name heur
"""

from __future__ import annotations

# --- path bootstrap: make the project root importable when this script is run
# from its subfolder, so `import configs` / `from rl_llm_multi ...` still resolve.
import os as _bootstrap_os, sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, _bootstrap_os.path.dirname(
    _bootstrap_os.path.dirname(_bootstrap_os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback if TensorBoard is unavailable."""

        def __init__(self, *_, **__) -> None:
            pass

        def add_scalar(self, *_, **__) -> None:
            pass

        def close(self) -> None:
            pass

from configs import (
    ACTION_BINS,
    DISTILL_DATASET_DIR,
    MAX_AGENTS,
    TEACHER_ACTIVATION,
    TEACHER_BATCH_SIZE,
    TEACHER_EARLY_STOP_PATIENCE,
    TEACHER_EPOCHS,
    TEACHER_HEUR_PATH,
    TEACHER_HIDDEN_DIMS,
    TEACHER_LABEL_SMOOTHING,
    TEACHER_LLM_PATH,
    TEACHER_LR,
    TEACHER_ORTHOGONAL_INIT,
    TEACHER_USE_LAYER_NORM,
    TEACHER_VAL_FRACTION,
    TEACHER_WEIGHT_DECAY,
)
from rl_llm_multi.mappo import device
from rl_llm_multi.teacher import (
    TeacherPolicy,
    load_trajectory_dataset,
    save_teacher,
)


def _split_indices(n: int, val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * float(val_fraction)))) if n > 1 else 0
    n_val = min(n_val, n - 1) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


@torch.no_grad()
def _accuracy(teacher: TeacherPolicy, obs: torch.Tensor, agent_idx: torch.Tensor, labels: torch.Tensor, batch: int = 4096) -> float:
    teacher.eval()
    correct = 0
    total = 0
    for start in range(0, obs.size(0), batch):
        o = obs[start:start + batch].to(device)
        a = agent_idx[start:start + batch].to(device)
        y = labels[start:start + batch].to(device)
        logits = teacher.action_logits(o, a)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct) / max(1, total)


def train_one(
    *,
    dataset_dir: Path,
    out_path: Path,
    name: str,
    args: argparse.Namespace,
) -> Dict:
    data = load_trajectory_dataset(dataset_dir)
    n = len(data)
    if n == 0:
        raise ValueError(f"Empty dataset at {dataset_dir}")
    obs_dim = data.obs_dim
    action_dim = len(ACTION_BINS)
    max_agents = int(data.meta.get("max_agents", MAX_AGENTS))

    print(f"\n=== Training teacher '{name}' from {dataset_dir} ===")
    print(f"  records={n}  obs_dim={obs_dim}  action_dim={action_dim}  max_agents={max_agents}")

    obs = torch.from_numpy(data.local_obs.astype(np.float32))
    agent_idx = torch.from_numpy(data.agent_index.astype(np.int64))
    labels = torch.from_numpy(data.action_idx.astype(np.int64))

    train_idx, val_idx = _split_indices(n, args.val_fraction, args.seed)
    train_ds = TensorDataset(obs[train_idx], agent_idx[train_idx], labels[train_idx])
    loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)

    teacher = TeacherPolicy(
        local_obs_dim=obs_dim,
        action_dim=action_dim,
        max_agents=max_agents,
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation=TEACHER_ACTIVATION,
        use_layer_norm=TEACHER_USE_LAYER_NORM,
        orthogonal_init=TEACHER_ORTHOGONAL_INIT,
    ).to(device)
    # Frozen input standardization fit on the training split.
    mean, std = data.input_stats()
    teacher.set_input_stats(torch.from_numpy(mean), torch.from_numpy(std))

    optimizer = torch.optim.Adam(
        teacher.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    # Majority-class baseline (what "always predict the most common action" gets).
    if val_idx.size > 0:
        val_labels = labels[val_idx]
        majority = int(torch.bincount(labels[train_idx], minlength=action_dim).argmax().item())
        majority_acc = float((val_labels == majority).float().mean().item())
    else:
        majority_acc = float("nan")

    # TensorBoard: per-epoch loss / val_acc under <out_path dir>/tensorboard/<name>
    # (or --tensorboard-dir/<name>). View with:  tensorboard --logdir teachers/tensorboard
    tb_root = Path(args.tensorboard_dir) if args.tensorboard_dir else Path(out_path).parent / "tensorboard"
    tb_dir = tb_root / name
    writer = SummaryWriter(str(tb_dir))
    print(f"  tensorboard -> {tb_dir}")

    best_val = -1.0
    best_state = None
    patience = 0
    for epoch in range(int(args.epochs)):
        teacher.train()
        epoch_loss = 0.0
        n_batches = 0
        for o, a, y in loader:
            o = o.to(device); a = a.to(device); y = y.to(device)
            logits = teacher.action_logits(o, a)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        if val_idx.size > 0:
            val_acc = _accuracy(teacher, obs[val_idx], agent_idx[val_idx], labels[val_idx])
        else:
            val_acc = _accuracy(teacher, obs[train_idx], agent_idx[train_idx], labels[train_idx])
        print(f"  epoch {epoch + 1:3d}/{args.epochs}  loss={epoch_loss / max(1, n_batches):.4f}  val_acc={val_acc:.4f}")
        step = epoch + 1
        writer.add_scalar("teacher/train_loss", epoch_loss / max(1, n_batches), step)
        writer.add_scalar("teacher/val_acc", val_acc, step)
        # Flat reference line so you can see val_acc rise above the lazy baseline.
        if majority_acc == majority_acc:  # not NaN
            writer.add_scalar("teacher/majority_val_acc", majority_acc, step)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= int(args.early_stop_patience):
                print(f"  early stop at epoch {epoch + 1} (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        teacher.load_state_dict(best_state)

    train_acc = _accuracy(teacher, obs[train_idx], agent_idx[train_idx], labels[train_idx])
    meta = {
        "teacher_name": name,
        "dataset_dir": str(dataset_dir),
        "dataset_meta": data.meta,
        "n_records": n,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "best_val_acc": float(best_val),
        "final_train_acc": float(train_acc),
        "majority_baseline_val_acc": majority_acc,
        "epochs_run": int(epoch + 1),
        "lr": float(args.lr),
        "label_smoothing": float(args.label_smoothing),
    }
    out_path = Path(out_path)
    save_teacher(teacher, out_path, meta=meta)
    print(f"  saved -> {out_path}")
    print(f"  best_val_acc={best_val:.4f}  train_acc={train_acc:.4f}  majority_val={majority_acc:.4f}")
    final_step = int(epoch + 1)
    writer.add_scalar("teacher/best_val_acc", float(best_val), final_step)
    writer.add_scalar("teacher/final_train_acc", float(train_acc), final_step)
    writer.close()
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--both", action="store_true",
                        help="Train teacher_llm from distill_data/real and "
                             "teacher_heur from distill_data/best_preview.")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--out-path", type=str, default=None)
    parser.add_argument("--name", type=str, default="teacher")
    parser.add_argument("--epochs", type=int, default=TEACHER_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TEACHER_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TEACHER_LR)
    parser.add_argument("--weight-decay", type=float, default=TEACHER_WEIGHT_DECAY)
    parser.add_argument("--label-smoothing", type=float, default=TEACHER_LABEL_SMOOTHING)
    parser.add_argument("--val-fraction", type=float, default=TEACHER_VAL_FRACTION)
    parser.add_argument("--early-stop-patience", type=int, default=TEACHER_EARLY_STOP_PATIENCE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard-dir", type=str, default=None,
                        help="Parent dir for TensorBoard logs; each teacher logs to "
                             "<dir>/<name>. Default: <out-path dir>/tensorboard/<name>.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Device: {device}")
    if args.both:
        train_one(
            dataset_dir=Path(DISTILL_DATASET_DIR) / "real",
            out_path=Path(TEACHER_LLM_PATH),
            name="llm",
            args=args,
        )
        train_one(
            dataset_dir=Path(DISTILL_DATASET_DIR) / "best_preview",
            out_path=Path(TEACHER_HEUR_PATH),
            name="heur",
            args=args,
        )
        return

    if not args.dataset_dir or not args.out_path:
        raise SystemExit("Provide --both, or both --dataset-dir and --out-path.")
    train_one(
        dataset_dir=Path(args.dataset_dir),
        out_path=Path(args.out_path),
        name=args.name,
        args=args,
    )


if __name__ == "__main__":
    main()
