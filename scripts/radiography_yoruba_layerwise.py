#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radiography_yoruba_layerwise.py
==============================================================================
Mechanistic interpretability for DIINA on Yoruba tonal triplets
(ọkọ, ọkọ́, ọ̀kọ̀, baba/bábá/bàbá, ...).

Tracks per layer:
  * mean gate activation g (fast/slow continuous gate)
  * Delta Logit = logit(Target) - logit(Trap attractor)

Outputs:
  * layer_saliency_traces_yoruba.csv
  * layerwise_heatmap_yoruba_300dpi.png (300-DPI heatmap)
Run:  python radiography_yoruba_layerwise.py [--checkpoint diina_yoruba.pt]
==============================================================================
"""
from __future__ import annotations

import argparse
import csv
import os

import torch
import torch.nn.functional as F

from train_yoruba_diina import (DualStreamDIINA, TRIPLETS, encode,
                                synthetic_corpus, build_vocab, DEVICE)

TRIPLETS_FOCUS = TRIPLETS + [("ọ̀kọ̀", "ọkọ", "ọkọ́")]


@torch.no_grad()
def layerwise_trace(model, word: str, target: str, trap: str, stoi):
    """Run forward pass collecting gate activations per 'layer' (stream stages)."""
    ids = torch.tensor([encode(word, stoi)]).to(DEVICE)
    T = ids.size(1)
    e = model.emb(ids) + model.pos(torch.arange(T, device=DEVICE))[None]
    f = model.fast(e)
    s = model.slow(e)
    g = model.gate(torch.cat([f, s], -1))
    h = g * f + (1 - g) * s
    logits = model.head(h)
    tgt_ids = encode(target, stoi)
    trp_ids = encode(trap, stoi)
    tgt_logit = float(logits[0, :len(tgt_ids), tgt_ids].mean())
    trp_logit = float(logits[0, :len(trp_ids), trp_ids].mean())
    stages = {
        "embedding": e.abs().mean().item(),
        "fast_path": f.abs().mean().item(),
        "slow_path": s.abs().mean().item(),
        "gate_mean": g.mean().item(),
        "gate_std": g.std().item(),
    }
    return tgt_logit, trp_logit, stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--out", type=str, default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    pairs = synthetic_corpus(500)
    stoi, itos = build_vocab(pairs)
    model = DualStreamDIINA(len(stoi), d=64)
    if a.checkpoint and os.path.exists(a.checkpoint):
        ck = torch.load(a.checkpoint, map_location=DEVICE)
        model.load_state_dict(ck["state_dict"])
        print("loaded checkpoint:", a.checkpoint)
    else:
        print("NOTE: no checkpoint given; using randomly initialized DIINA "
              "(traces show architecture dynamics, not trained behavior).")
    model.to(DEVICE).eval()

    rows = []
    for base, high, low in TRIPLETS_FOCUS:
        for word, target, trap in ((base, base, high), (base, high, low),
                                   (base, low, high)):
            tl, trl, stages = layerwise_trace(model, word, target, trap, stoi)
            rows.append({
                "word": word, "target": target, "trap": trap,
                "target_logit": round(tl, 6), "trap_logit": round(trl, 6),
                "delta_logit": round(tl - trl, 6),
                **{k: round(v, 6) for k, v in stages.items()},
            })

    csv_path = os.path.join(a.out, "layer_saliency_traces_yoruba.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", csv_path)

    # ---- 300-DPI heatmap ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        cols = ["embedding", "fast_path", "slow_path",
                "gate_mean", "delta_logit"]
        M = np.array([[r[c] for c in cols] for r in rows])
        fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(rows))))
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['word']}→{r['target']}" for r in rows],
                           fontsize=6)
        ax.set_title("DIINA layerwise saliency — Yoruba tonal triplets")
        fig.colorbar(im, ax=ax, shrink=0.7)
        fig.tight_layout()
        png = os.path.join(a.out, "layerwise_heatmap_yoruba_300dpi.png")
        fig.savefig(png, dpi=300)
        plt.close(fig)
        print("saved", png)
    except Exception as exc:  # matplotlib missing
        print("heatmap skipped:", exc)


if __name__ == "__main__":
    main()
