#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_yoruba_diina.py
==============================================================================
Full PyTorch neural model for Yoruba diacritic & tonal modeling.

DualStreamDIINA:
  * Fast Path  : FeedForward / GRU  (fast, tone-sensitive stream)
  * Slow Path  : Self-Attention + Deep Linear (long-range phonotactics)
  * Continuous Gating: sigmoid gate with inhibition + attractor penalty
Training:
  * AdamW + cosine annealing LR schedule
  * Synthetic Yoruba corpus (tonal triplets) or loaded corpus file
  * Evaluation on minimal pairs (ọkọ vs ọkọ́ vs ọ̀kọ̀ ...)
Run:  python train_yoruba_diina.py --epochs 20 [--corpus path.txt]
==============================================================================
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Synthetic Yoruba corpus
# ---------------------------------------------------------------------------
TRIPLETS = [
    ("ọkọ", "ọkọ́", "ọkọ̀"), ("baba", "bábá", "bàbá"), ("igba", "ígbá", "ìgbà"),
    ("ọmọ", "ọmọ́", "ọmọ̀"), ("ilé", "ilẹ̀", "ìlé"), ("ara", "ára", "àrà"),
    ("òwe", "owé", "òwè"), ("èdè", "édè", "èdẹ"), ("oko", "ọkọ́", "ọ̀kọ"),
]
SYLL_PREFIX = ["mo", "o", "a", "ó", "ọ", "kí", "ní", "sì"]


def synthetic_corpus(n: int = 2000, seed: int = 7) -> List[Tuple[str, str]]:
    """Returns (flattened_input, accented_target) pairs."""
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        base, high, low = rng.choice(TRIPLETS)
        tgt = rng.choice([base, high, low])
        ctx = rng.choice(SYLL_PREFIX) + " "
        data.append((ctx + tgt.replace("\u0300", "").replace("\u0301", ""), tgt))
    return data


def load_corpus(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            flat = "".join(c for c in line
                           if c not in ("\u0300", "\u0301", "\u0323"))
            pairs.append((flat, line))
    return pairs


def build_vocab(pairs: List[Tuple[str, str]]) -> Tuple[dict, dict]:
    chars = sorted({c for a, b in pairs for c in (a + b)})
    stoi = {c: i + 2 for i, c in enumerate(chars)}
    stoi["<pad>"] = 0
    stoi["<unk>"] = 1
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(s: str, stoi: dict) -> List[int]:
    return [stoi.get(c, 1) for c in s]


# ---------------------------------------------------------------------------
# DualStreamDIINA module
# ---------------------------------------------------------------------------
class FastPath(nn.Module):
    """FeedForward + GRU fast stream (surface tone cues)."""

    def __init__(self, d: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.gru = nn.GRU(d, d, batch_first=True)

    def forward(self, x):
        h = self.ff(x)
        out, _ = self.gru(h)
        return out


class SlowPath(nn.Module):
    """Self-attention + deep linear slow stream (long-range structure)."""

    def __init__(self, d: int, layers: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        h = a
        for lin, norm in zip(self.linears, self.norms):
            h = norm(h + F.gelu(lin(h)))
        return h


class DualStreamDIINA(nn.Module):
    def __init__(self, vocab_size: int, d: int = 64, attnor_penalty: float = 0.1):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos = nn.Embedding(512, d)
        self.fast = FastPath(d)
        self.slow = SlowPath(d)
        # continuous gating: sigmoid inhibition between streams
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.head = nn.Linear(d, vocab_size)
        self.attractor_penalty = attnor_penalty

    def forward(self, x, return_gates: bool = False):
        T = x.size(1)
        e = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
        f = self.fast(e)
        s = self.slow(e)
        g = self.gate(torch.cat([f, s], dim=-1))     # in (0,1), inhibition
        h = g * f + (1.0 - g) * s
        logits = self.head(h)
        if return_gates:
            return logits, g
        return logits

    def attractor_loss(self, g: torch.Tensor) -> torch.Tensor:
        """Penalise gate saturation (trapped in fixed-point attractors)."""
        return self.attractor_penalty * (g * (1 - g)).mean() * -1.0 + \
               self.attractor_penalty * (torch.abs(g - 0.5) ** 2).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def make_batches(pairs, stoi, bs=32, shuffle=True):
    idx = list(range(len(pairs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [pairs[j] for j in idx[i:i + bs]]
        src = [encode(a, stoi) for a, _ in chunk]
        tgt = [encode(b, stoi) for _, b in chunk]
        L = max(max(map(len, src)), max(map(len, tgt)))
        pad = lambda seq: [s + [0] * (L - len(s)) for s in seq]
        yield (torch.tensor(pad(src)), torch.tensor(pad(tgt)))


def train(model, pairs, stoi, epochs=20, lr=3e-4, bs=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(DEVICE)
    for ep in range(1, epochs + 1):
        tot, nb = 0.0, 0
        for src, tgt in make_batches(pairs, stoi, bs):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            logits, g = model(src, return_gates=True)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 tgt.reshape(-1), ignore_index=0)
            loss = ce + model.attractor_loss(g)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += ce.item(); nb += 1
        sched.step()
        print(f"epoch {ep:03d}  CE={tot/max(nb,1):.4f}  lr={sched.get_last_lr()[0]:.2e}")
    return model


@torch.no_grad()
def eval_minimal_pairs(model, stoi, itos) -> float:
    """Accuracy choosing the correctly accented variant among triplet members."""
    model.eval()
    correct = total = 0
    for base, high, low in TRIPLETS:
        for target in (base, high, low):
            flat = base  # all variants share flattened form
            src = torch.tensor([encode(flat, stoi)]).to(DEVICE)
            logits, _ = model(src, return_gates=True)
            # score each variant by mean log-prob of its characters
            scores = []
            for cand in (base, high, low):
                ids = encode(cand, stoi)
                lp = F.log_softmax(logits[0, :len(ids)], -1)
                scores.append(sum(lp[k, ids[k]].item() for k in range(len(ids))))
            pred = (base, high, low)[int(torch.tensor(scores).argmax())]
            correct += int(pred == target); total += 1
    acc = correct / total
    print(f"minimal-pair accuracy: {acc:.3f}")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=None,
                    help="optional UTF-8 text corpus of Yoruba sentences")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n-synthetic", type=int, default=2000)
    ap.add_argument("--out", type=str, default="diina_yoruba.pt")
    a = ap.parse_args()

    pairs = load_corpus(a.corpus) if a.corpus else synthetic_corpus(a.n_synthetic)
    stoi, itos = build_vocab(pairs)
    model = DualStreamDIINA(len(stoi), d=a.d)
    model = train(model, pairs, stoi, epochs=a.epochs)
    acc = eval_minimal_pairs(model, stoi, itos)
    torch.save({"state_dict": model.state_dict(),
                "stoi": stoi, "minimal_pair_acc": acc}, a.out)
    print("saved checkpoint ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
