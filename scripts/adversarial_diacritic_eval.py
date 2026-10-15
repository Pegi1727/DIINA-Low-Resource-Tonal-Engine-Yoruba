#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adversarial_diacritic_eval.py
==============================================================================
Simulates Common Crawl / OSCAR corpus-induced *tonal blindness* noise:
  * diacritic stripping      (combining marks removed)
  * random diacritic substitution (grave<->acute swaps at rate eta)
  * ASCII flattening         (underdot vowels -> plain ASCII)

Compares robustness curves (accuracy vs noise level eta in [0,1]) of
DIINA vs Transformer vs Mamba baselines. Exports robustness_curves CSV+PNG.
Run:  python adversarial_diacritic_eval.py --out .
==============================================================================
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random

import numpy as np
import torch

from benchmark_yoruba_stress import (build_stress_dataset, make_transformer,
                                     make_diina, make_mamba, strip_diacritics)

# ---------------------------------------------------------------------------
# Noise operators (corpus-induced tonal blindness)
# ---------------------------------------------------------------------------
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def strip_marks(w: str) -> str:
    return strip_diacritics(w)


def substitute_marks(w: str, eta: float, rng: random.Random) -> str:
    out = []
    for ch in w:
        if ch == "\u0300" and rng.random() < eta:
            out.append("\u0301")
        elif ch == "\u0301" and rng.random() < eta:
            out.append("\u0300")
        else:
            out.append(ch)
    return "".join(out)


def corrupt(word: str, eta: float, rng: random.Random, mode: str) -> str:
    if mode == "strip":
        return strip_marks(word) if rng.random() < eta else word
    if mode == "substitute":
        return substitute_marks(word, eta, rng)
    if mode == "ascii":
        return strip_diacritics(word) if rng.random() < eta else word
    raise ValueError(mode)


# ---------------------------------------------------------------------------
# Robustness evaluation
# ---------------------------------------------------------------------------
def build_vocab_from_items(items):
    chars = sorted({c for it in items for c in
                    (it["context"] + it["target"] + it["trap"])})
    return {c: i + 1 for i, c in enumerate(chars)} | {"<unk>": 0}


@torch.no_grad()
def accuracy(model, items, vocab, eta, rng, mode) -> float:
    correct = 0
    for it in items:
        tgt = corrupt(it["target"], eta, rng, mode)
        ids = torch.tensor([[vocab.get(ch, 0) for ch in it["context"][:64]]])
        zt = float(model(ids)[0, -1, vocab.get(tgt, 0)])
        zr = float(model(ids)[0, -1, vocab.get(it["trap"], 0)])
        correct += int(zt > zr)
    return correct / len(items)


def run(n_items: int = 1000, seed: int = 2022, out: str = "."):
    os.makedirs(out, exist_ok=True)
    items = build_stress_dataset(n_items, seed)
    vocab = build_vocab_from_items(items)
    V = len(vocab)
    torch.manual_seed(seed)
    models = {"DIINA": make_diina(V), "Transformer": make_transformer(V),
              "Mamba": make_mamba(V)}
    rng_master = random.Random(seed)

    curves = {}
    for mname, model in models.items():
        curves[mname] = {}
        for mode in ("strip", "substitute", "ascii"):
            rng = random.Random(seed + hash(mode) % 9973)
            curves[mname][mode] = [accuracy(model, items, vocab, eta, rng, mode)
                                   for eta in NOISE_LEVELS]
            print(mname, mode, curves[mname][mode])

    csv_path = os.path.join(out, "adversarial_robustness_curves.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "noise_mode"] + [f"eta={e}" for e in NOISE_LEVELS])
        for mname in curves:
            for mode in curves[mname]:
                w.writerow([mname, mode] +
                           [round(v, 4) for v in curves[mname][mode]])
    print("saved", csv_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, mode in zip(axes, ("strip", "substitute", "ascii")):
            for mname, style in zip(curves, ("-", "--", ":")):
                ax.plot(NOISE_LEVELS, curves[mname][mode], style,
                        marker="o", label=mname)
            ax.set_title(mode); ax.set_xlabel("noise η"); ax.grid(alpha=.3)
        axes[0].set_ylabel("accuracy")
        axes[0].legend()
        fig.suptitle("Yoruba tonal-blindness robustness: DIINA vs baselines")
        fig.tight_layout()
        png = os.path.join(out, "adversarial_robustness_curves_300dpi.png")
        fig.savefig(png, dpi=300)
        print("saved", png)
    except Exception as exc:
        print("plot skipped:", exc)
    return curves


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--out", type=str, default=".")
    a = ap.parse_args()
    run(a.n, a.seed, a.out)
