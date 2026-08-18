"""Stage 3 -- train the student against the fused teacher target.

Deliberately not built on TRL's `GKDTrainer`, despite that being the reference
implementation of on-policy distillation.  `GKDTrainer` keeps the teacher resident in
the training process and generates on-policy samples with HF `generate`.  Both of
those are exactly what the staged design exists to avoid: co-resident teachers cap
teacher size at whatever is left over after the optimiser state, and HF `generate` is
roughly an order of magnitude slower than vLLM at this batch size.  The loss in
`mopd.loss` follows the same generalised-JSD convention as TRL so numbers stay
comparable, and `tests/test_loss.py` checks that against TRL when it is installed.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.store import MOPDDataset, collate, gather_pred_logits
from .fusion import fuse_teachers
from .loss import failure_aware_weights, jsd_loss
from .router import route, routing_entropy


@dataclass
class TrainConfig:
    student: str
    rollout_path: str
    teacher_paths: dict[str, str]           # role -> parquet path
    out_dir: str

    # --- the experiment matrix lives in these four fields ---
    router: str = "confidence"              # uniform | single | oracle | confidence | ...
    fusion_mode: str = "arithmetic"         # arithmetic | geometric
    beta: float = 0.5                       # 0 reverse KL .. 1 forward KL
    failure_aware: bool = True

    router_temperature: float = 0.5
    single_index: int = 0
    temperature: float = 1.0
    tail_uniform: bool = True

    lr: float = 1e-5
    epochs: int = 1
    batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_len: int = 1280
    gradient_checkpointing: bool = True
    optim: str = "adamw"                    # adamw | adamw_8bit
    seed: int = 0
    log_every: int = 10
    save_every: int | None = None
    tags: list[str] = field(default_factory=list)


def train(cfg: TrainConfig) -> Path:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    torch.manual_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    roles = list(cfg.teacher_paths)
    tok = AutoTokenizer.from_pretrained(cfg.student)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    ds = MOPDDataset(cfg.rollout_path, cfg.teacher_paths, max_len=cfg.max_len)
    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: collate(b, pad_id=pad_id, roles=roles),
        drop_last=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.student, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.cuda().train()
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    vocab_size = model.get_output_embeddings().weight.shape[0]

    optimizer = _build_optimizer(model, cfg)
    total_steps = max(1, (len(dl) // cfg.grad_accum) * cfg.epochs)
    sched = get_cosine_schedule_with_warmup(
        optimizer, int(total_steps * cfg.warmup_ratio), total_steps
    )

    logger = _Logger(out_dir, cfg)
    step, t0 = 0, time.time()

    for epoch in range(cfg.epochs):
        for it, batch in enumerate(dl):
            stats = _train_step(model, batch, cfg, roles, vocab_size)
            (stats["loss"] / cfg.grad_accum).backward()

            if (it + 1) % cfg.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % cfg.log_every == 0:
                    logger.log(
                        {
                            "step": step,
                            "epoch": epoch,
                            "loss": float(stats["loss"]),
                            "route_entropy": float(stats["route_entropy"]),
                            **{f"w/{r}": float(v) for r, v in stats["weights"].items()},
                            "grad_norm": float(gnorm),
                            "lr": sched.get_last_lr()[0],
                            "tok_per_s": stats["n_tokens"] * cfg.log_every / (time.time() - t0),
                        }
                    )
                    t0 = time.time()

                if cfg.save_every and step % cfg.save_every == 0:
                    _save(model, tok, out_dir / f"step-{step}")

    _save(model, tok, out_dir / "final")
    logger.close()
    return out_dir / "final"


def _train_step(model, batch, cfg: TrainConfig, roles: list[str], vocab_size: int) -> dict:
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)
    loss_mask = batch["loss_mask"].to(device)
    signals = [s.to(device) for s in batch["signals"]]
    n_pred = loss_mask.shape[1]

    weights = route(
        cfg.router,
        roles,
        mean_logprobs=batch["mean_logprobs"],
        domains=batch["domains"],
        single_index=cfg.single_index,
        temperature=cfg.router_temperature,
        device=device,
    )

    target = fuse_teachers(
        signals,
        weights,
        vocab_size=vocab_size,
        mode=cfg.fusion_mode,
        tail_uniform=cfg.tail_uniform,
    )

    logits = model(input_ids=input_ids, attention_mask=attn).logits[:, :-1]
    logits = gather_pred_logits(logits, batch["pred_offset"].to(device), n_pred)

    token_w = None
    if cfg.failure_aware:
        token_w = failure_aware_weights(
            logits.detach(), target, loss_mask, batch["reward"].to(device)
        )

    loss = jsd_loss(
        logits, target, loss_mask, beta=cfg.beta,
        temperature=cfg.temperature, token_weights=token_w,
    )

    return {
        "loss": loss,
        "route_entropy": routing_entropy(weights).mean(),
        "weights": dict(
            zip(roles, weights.reshape(-1, len(roles)).mean(0).tolist(), strict=True)
        ),
        "n_tokens": int(loss_mask.sum()),
    }


def _build_optimizer(model, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optim == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)


def _save(model, tok, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)
    print(f"[train] saved -> {path}")


class _Logger:
    """stdout + JSONL always; W&B when it is installed and configured."""

    def __init__(self, out_dir: Path, cfg: TrainConfig):
        self.f = (out_dir / "log.jsonl").open("a")
        self.wandb = None
        try:
            import wandb

            self.wandb = wandb.init(
                project="mopd", config=asdict(cfg), tags=cfg.tags, dir=str(out_dir)
            )
        except Exception as e:
            print(f"[train] W&B disabled ({type(e).__name__}: {e}); logging to JSONL only")

    def log(self, row: dict) -> None:
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()
        pretty = " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}"
            for k, v in row.items()
        )
        print(f"[train] {pretty}", flush=True)
        if self.wandb:
            self.wandb.log(row, step=row["step"])

    def close(self) -> None:
        self.f.close()
        if self.wandb:
            self.wandb.finish()
