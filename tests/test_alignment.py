"""Alignment is the failure mode that produces a plausible-looking loss curve and a
model that learned nothing. These tests pin the index arithmetic down.

Convention, fixed once here and relied on everywhere else:
    logits[:, t] predicts input_ids[:, t + 1]
    a row with prompt length P and total length N has supervised prediction indices
    t in [P - 1, N - 2], i.e. exactly N - P of them, one per completion token.
"""

import torch

from mopd.data.store import collate, gather_pred_logits


def fake_item(traj_id, prompt_len, n_pred, k=4, vocab=32, reward=1.0):
    n = prompt_len + n_pred
    return {
        "traj_id": traj_id,
        "domain": "math",
        "input_ids": torch.arange(n) % vocab,
        "prompt_len": prompt_len,
        "n_pred": n_pred,
        "reward": reward,
        "signals": {
            "t1": {
                "topk_ids": torch.zeros(n_pred, k, dtype=torch.long),
                "topk_logprobs": torch.full((n_pred, k), -1.0),
                "tail_mass": torch.zeros(n_pred),
                "mean_logprob": -0.5,
            }
        },
    }


def test_gather_pred_logits_picks_the_completion_window():
    # encode position in the logits so we can read back which index was selected
    b, t, v = 2, 10, 3
    logits = torch.arange(t, dtype=torch.float32).view(1, t, 1).expand(b, t, v).contiguous()

    prompt_lens = torch.tensor([4, 7])
    pred_offset = prompt_lens - 1
    got = gather_pred_logits(logits, pred_offset, n_pred=3)

    assert torch.equal(got[0, :, 0], torch.tensor([3.0, 4.0, 5.0]))  # predicts tokens 4,5,6
    assert torch.equal(got[1, :, 0], torch.tensor([6.0, 7.0, 8.0]))  # predicts tokens 7,8,9


def test_last_supervised_index_is_n_minus_two():
    """The final prediction must be the one that emits the last completion token."""
    n, p, v = 9, 3, 2
    logits = torch.arange(n - 1, dtype=torch.float32).view(1, n - 1, 1).expand(1, n - 1, v)
    got = gather_pred_logits(logits.contiguous(), torch.tensor([p - 1]), n_pred=n - p)
    assert float(got[0, -1, 0]) == n - 2


def test_collate_masks_only_real_predictions():
    batch = [fake_item(0, prompt_len=3, n_pred=5), fake_item(1, prompt_len=6, n_pred=2)]
    out = collate(batch, pad_id=0, roles=["t1"])

    assert out["input_ids"].shape == (2, 8)
    assert out["loss_mask"].shape == (2, 5)
    assert out["loss_mask"][0].tolist() == [True] * 5
    assert out["loss_mask"][1].tolist() == [True, True, False, False, False]
    assert out["pred_offset"].tolist() == [2, 5]
    assert out["attention_mask"].sum(1).tolist() == [8, 8]


def test_collate_pads_teacher_signals_without_leaking():
    batch = [fake_item(0, 3, 5), fake_item(1, 6, 2)]
    out = collate(batch, pad_id=0, roles=["t1"])
    sig = out["signals"][0]
    assert sig.topk_logprobs.shape == (2, 5, 4)
    # padded rows keep the -30 floor, and the mask excludes them anyway
    assert torch.all(sig.topk_logprobs[1, 2:] == -30.0)
    assert torch.all(sig.topk_logprobs[1, :2] == -1.0)


def test_collate_stacks_router_inputs():
    batch = [fake_item(0, 3, 5, reward=1.0), fake_item(1, 6, 2, reward=0.0)]
    out = collate(batch, pad_id=0, roles=["t1"])
    assert out["mean_logprobs"].shape == (2, 1)
    assert out["reward"].tolist() == [1.0, 0.0]
    assert out["domains"] == ["math", "math"]


def test_end_to_end_shapes_line_up():
    """collate -> gather -> fuse -> loss, on the shapes the trainer actually uses."""
    from mopd.fusion import fuse_teachers
    from mopd.loss import jsd_loss

    vocab = 32
    batch = [fake_item(0, 3, 5, vocab=vocab), fake_item(1, 6, 2, vocab=vocab)]
    out = collate(batch, pad_id=0, roles=["t1"])
    n_pred = out["loss_mask"].shape[1]

    full_logits = torch.randn(2, out["input_ids"].shape[1] - 1, vocab)
    logits = gather_pred_logits(full_logits, out["pred_offset"], n_pred)
    assert logits.shape == (2, n_pred, vocab)

    target = fuse_teachers(out["signals"], torch.ones(2, 1), vocab_size=vocab)
    loss = jsd_loss(logits, target, out["loss_mask"], beta=0.5)
    assert torch.isfinite(loss) and loss > 0
