"""On-disk format for trajectories and teacher signals.

The three pipeline stages never hold two models in memory at once, so they hand off
through Parquet:

    stage 1  rollout.parquet          student trajectories + verifier reward
    stage 2  teacher-<role>.parquet   that role's top-K logprobs for every trajectory
    stage 3  reads both, streams batches

Teacher logprobs are stored flattened (``n_pred * k`` per row) with `n_pred` recorded
alongside, because Parquet list columns are cheap but nested list columns are not.

Sizing, for the reference config (24k trajectories x 512 tokens x K=64):
    rollout      ~0.2 GB
    per teacher  ~3.1 GB   (float32; halve it by setting store_fp16=True)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

K_DEFAULT = 64


def _pa():
    import pyarrow as pa

    return pa


def rollout_schema():
    pa = _pa()
    return pa.schema(
        [
            ("traj_id", pa.int32()),
            ("uid", pa.string()),
            ("domain", pa.string()),
            ("prompt_len", pa.int32()),
            ("input_ids", pa.list_(pa.int32())),
            ("completion", pa.string()),
            ("reward", pa.float32()),
            ("iteration", pa.int32()),
        ]
    )


def teacher_schema(fp16: bool = False):
    pa = _pa()
    val = pa.float16() if fp16 else pa.float32()
    return pa.schema(
        [
            ("traj_id", pa.int32()),
            ("n_pred", pa.int32()),
            ("k", pa.int32()),
            ("topk_ids", pa.list_(pa.int32())),
            ("topk_logprobs", pa.list_(val)),
            ("tail_mass", pa.list_(val)),
            ("mean_logprob", pa.float32()),
        ]
    )


def write_table(path: str | Path, rows: list[dict], schema) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path


def read_table(path: str | Path) -> dict[int, dict]:
    """Read a Parquet shard into a traj_id -> row dict."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    return {row["traj_id"]: row for row in table.to_pylist()}


class MOPDDataset(torch.utils.data.Dataset):
    """Joins student trajectories with every teacher's signal for that trajectory."""

    def __init__(
        self,
        rollout_path: str | Path,
        teacher_paths: dict[str, str | Path],
        max_len: int | None = None,
    ):
        self.rollouts = read_table(rollout_path)
        self.teachers = {role: read_table(p) for role, p in teacher_paths.items()}
        self.roles = list(teacher_paths)
        self.max_len = max_len
        self.ids = sorted(
            tid for tid in self.rollouts
            if all(tid in t for t in self.teachers.values())
        )
        missing = len(self.rollouts) - len(self.ids)
        if missing:
            print(f"[store] dropping {missing} trajectories without full teacher coverage")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        tid = self.ids[i]
        r = self.rollouts[tid]
        ids = torch.tensor(r["input_ids"], dtype=torch.long)
        prompt_len = int(r["prompt_len"])
        n_pred = len(ids) - prompt_len  # supervised next-token predictions

        sig = {}
        for role in self.roles:
            t = self.teachers[role][tid]
            k, np_ = int(t["k"]), int(t["n_pred"])
            sig[role] = {
                "topk_ids": torch.tensor(
                    np.asarray(t["topk_ids"], dtype=np.int64).reshape(np_, k)
                ),
                "topk_logprobs": torch.tensor(
                    np.asarray(t["topk_logprobs"], dtype=np.float32).reshape(np_, k)
                ),
                "tail_mass": torch.tensor(
                    np.asarray(t["tail_mass"], dtype=np.float32)
                ),
                "mean_logprob": float(t["mean_logprob"]),
            }
            n_pred = min(n_pred, np_)

        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[: self.max_len]
            n_pred = min(n_pred, self.max_len - prompt_len)

        n_pred = max(n_pred, 0)
        for role in self.roles:
            for key in ("topk_ids", "topk_logprobs"):
                sig[role][key] = sig[role][key][:n_pred]
            sig[role]["tail_mass"] = sig[role]["tail_mass"][:n_pred]

        return {
            "traj_id": tid,
            "domain": r["domain"],
            "input_ids": ids[: prompt_len + n_pred],
            "prompt_len": prompt_len,
            "n_pred": n_pred,
            "reward": float(r["reward"]),
            "signals": sig,
        }


def collate(batch: list[dict], pad_id: int, roles: list[str]) -> dict:
    """Right-pad to the batch max. Returns tensors plus per-role SparseLogprobs parts.

    `loss_mask` is aligned to *prediction* positions: index t of the shifted logits
    predicts input_ids[t + 1], so the first supervised index is prompt_len - 1.
    """
    from ..fusion import SparseLogprobs

    b = len(batch)
    lens = [len(x["input_ids"]) for x in batch]
    npred = [x["n_pred"] for x in batch]
    tmax, pmax = max(lens), max([*npred, 1])
    k = batch[0]["signals"][roles[0]]["topk_ids"].shape[-1]

    input_ids = torch.full((b, tmax), pad_id, dtype=torch.long)
    attn = torch.zeros((b, tmax), dtype=torch.long)
    loss_mask = torch.zeros((b, pmax), dtype=torch.bool)
    pred_offset = torch.zeros(b, dtype=torch.long)

    topk_ids = torch.zeros((b, pmax, k), dtype=torch.long)
    topk_lp = torch.full((b, pmax, k), -30.0)
    tail = torch.zeros((b, pmax))
    sig = {r: (topk_ids.clone(), topk_lp.clone(), tail.clone()) for r in roles}
    mean_lp = torch.zeros((b, len(roles)))

    for i, x in enumerate(batch):
        n, p = lens[i], npred[i]
        input_ids[i, :n] = x["input_ids"]
        attn[i, :n] = 1
        loss_mask[i, :p] = True
        pred_offset[i] = x["prompt_len"] - 1
        for j, role in enumerate(roles):
            s = x["signals"][role]
            sig[role][0][i, :p] = s["topk_ids"]
            sig[role][1][i, :p] = s["topk_logprobs"]
            sig[role][2][i, :p] = s["tail_mass"]
            mean_lp[i, j] = s["mean_logprob"]

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "loss_mask": loss_mask,
        "pred_offset": pred_offset,
        "reward": torch.tensor([x["reward"] for x in batch]),
        "domains": [x["domain"] for x in batch],
        "mean_logprobs": mean_lp,
        "signals": [SparseLogprobs(*sig[r]) for r in roles],
    }


def gather_pred_logits(logits: torch.Tensor, pred_offset: torch.Tensor,
                       n_pred: int) -> torch.Tensor:
    """Slice each row's supervised prediction window out of [B, T, V] logits.

    Row i's supervised predictions start at `pred_offset[i]`, but rows have different
    prompt lengths, so a single slice will not do.
    """
    b, t, v = logits.shape
    idx = pred_offset.view(b, 1) + torch.arange(n_pred, device=logits.device).view(1, -1)
    idx = idx.clamp(max=t - 1)
    return logits.gather(1, idx.unsqueeze(-1).expand(b, n_pred, v))
