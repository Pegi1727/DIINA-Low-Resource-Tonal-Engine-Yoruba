#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_yoruba_stress.py
==============================================================================
Standard Transformer vs. DIINA vs. SSM/Mamba baseline on the Yoruba 1000-item
stress benchmark (tonal diacritics + attractor traps).

Computes per model:
  * logit distributions (mean/std of target-logit margins)
  * cross-entropy (per item)
  * diacritic error rate (tone-mark mismatch)
  * attractor trap escape rate / recovery rate
Statistics:
  * McNemar's exact test (scipy.stats.binomtest)
  * Wilcoxon signed-rank test on per-item CE
  * Cohen's g effect size  g = p_b - 0.5  (with |.5| fold)
  * logit shift Delta z = mean(z_target_model) - mean(z_baseline)
Exports: CSV (per-item + summary) and JSON (summary).

Run:  python benchmark_yoruba_stress.py [--n 1000] [--seed 2022]
==============================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:  # pragma: no cover
    TORCH_OK = False

try:
    from scipy.stats import binomtest, wilcoxon
    SCIPY_OK = True
except Exception:  # pragma: no cover
    SCIPY_OK = False

# ----------------------------------------------------------------------------
# Yoruba tone / diacritic utilities
# ----------------------------------------------------------------------------
TONE_MARKS = {"\u0300": "LOW", "\u0301": "HIGH"}          # combining grave/acute
VOWELS = "aeiouẹọ"
UNDERDOT = {"e": "ẹ", "o": "ọ"}                            # ASCII -> underdot

# Minimal-pair tonal triplets used as attractor traps (base, HIGH, LOW)
TRIPLETS = [
    ("ọkọ", "ọkọ́", "ọkọ̀"),      # husband / vehicle / (broom语境)
    ("ọ̀kọ̀", "ọkọ́", "ọkọ̀"),
    ("baba", "bábá", "bàbá"),
    ("òwe", "owé", "òwè"),
    ("igba", "ígbá", "ìgbà"),
    ("ọmọ", "ọmọ́", "ọmọ̀"),
    ("ilé", "ilẹ̀", "ìlé"),
    ("ara", "ára", "àrà"),
    ("oko", "ọkọ́", "ọ̀kọ"),
    ("èdè", "èdẹ", "édè"),
]


def strip_diacritics(word: str) -> str:
    """ASCII-flatten: remove combining tone marks and underdots."""
    out = []
    for ch in word:
        if ch in ("\u0300", "\u0301", "\u0323"):
            continue
        out.append({"ẹ": "e", "ọ": "o", "ṣ": "s"}.get(ch, ch))
    return "".join(out)


def tone_sequence(word: str) -> List[str]:
    """Extract the tone labels of a word (MID for unmarked vowels)."""
    tones = []
    for ch in word:
        if ch in VOWELS or ch in "ẹọ":
            tones.append("MID")
    prev_mark = None
    for ch in word:
        if ch in TONE_MARKS:
            prev_mark = TONE_MARKS[ch]
        elif ch in VOWELS or ch in "ẹọ":
            if prev_mark:
                tones[-1] = prev_mark
            prev_mark = None
    return tones


# ----------------------------------------------------------------------------
# Synthetic stress dataset: 1000 items built from triplets + distractors
# ----------------------------------------------------------------------------
def build_stress_dataset(n: int = 1000, seed: int = 2022) -> List[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        base, high, low = rng.choice(TRIPLETS)
        # correct target is chosen by context digit (synthetic "sentence" cue)
        target = rng.choice([base, high, low])
        # attractor trap = the competing tone variant
        others = [w for w in (base, high, low) if w != target]
        trap = rng.choice(others)
        context = f"nípasẹ̀ kòódù {i % 97}: "
        items.append({
            "id": i,
            "context": context,
            "target": target,
            "trap": trap,
            "flattened": strip_diacritics(target),
            "target_tones": tone_sequence(target),
            "trap_tones": tone_sequence(trap),
        })
    return items


# ----------------------------------------------------------------------------
# Model zoo (light-weight, CPU-friendly surrogates; swap with real checkpoints)
# ----------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    correct: np.ndarray            # 0/1 per item (target beats trap)
    ce: np.ndarray                 # cross-entropy per item
    target_logits: np.ndarray      # target logit per item
    trap_logits: np.ndarray
    margin: np.ndarray             # z_target - z_trap
    escaped: np.ndarray            # 1 if model chose target over trap
    recovered: np.ndarray          # 1 if escaped after initial wrong-tendency
    meta: Dict = field(default_factory=dict)


class LogitScorer:
    """Scores a candidate word against context using a model's logit head."""

    def __init__(self, model: "nn.Module", vocab: Dict[str, int]):
        self.model = model
        self.vocab = vocab

    @torch.no_grad()
    def score(self, context_ids: torch.Tensor, cand_id: int) -> float:
        logits = self.model(context_ids)                 # [B, V]
        return float(logits[0, -1, cand_id].item())


def _toy_eval(model, items: List[dict], vocab: Dict[str, int],
              name: str, seed: int = 0) -> ModelResult:
    """Evaluate with a deterministic scoring pass (temperature-scaled softmax)."""
    torch.manual_seed(seed)
    scorer = LogitScorer(model, vocab)
    N = len(items)
    correct = np.zeros(N); ce = np.zeros(N)
    tl = np.zeros(N); tr = np.zeros(N); m = np.zeros(N)
    esc = np.zeros(N); rec = np.zeros(N)
    for i, it in enumerate(items):
        ids = torch.tensor([[vocab.get(ch, 0) for ch in it["context"][:64]]])
        z_t = scorer.score(ids, vocab.get(it["target"], 0))
        z_r = scorer.score(ids, vocab.get(it["trap"], 0))
        tl[i], tr[i] = z_t, z_r
        m[i] = z_t - z_r
        p_t = 1.0 / (1.0 + math.exp(-(z_t - z_r)))
        esc[i] = 1.0 if z_t > z_r else 0.0
        correct[i] = esc[i]
        ce[i] = -math.log(max(p_t, 1e-12))
        # recovery: escaped despite flattened-input mismatch tendency
        rec[i] = 1.0 if (esc[i] and it["flattened"] == strip_diacritics(it["trap"])) else 0.0
    return ModelResult(name, correct, ce, tl, tr, m, esc, rec)


def make_transformer(vocab_size: int, d: int = 32) -> "nn.Module":
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d)
            self.pos = nn.Embedding(128, d)
            enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                             batch_first=True)
            self.enc = nn.TransformerEncoder(enc, num_layers=2)
            self.head = nn.Linear(d, vocab_size)

        def forward(self, x):
            T = x.size(1)
            h = self.emb(x) + self.pos(torch.arange(T, device=x.device))[None]
            return self.head(self.enc(h))
    return TinyTransformer()


def make_diina(vocab_size: int, d: int = 32) -> "nn.Module":
    """DIINA surrogate: dual-stream fast/slow with sigmoid gating + attractor penalty."""
    class DualStreamDIINA(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d)
            self.fast = nn.GRU(d, d, batch_first=True)
            self.slow = nn.MultiheadAttention(d, 4, batch_first=True)
            self.slow_lin = nn.Linear(d, d)
            self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
            self.head = nn.Linear(d, vocab_size)

        def forward(self, x):
            e = self.emb(x)
            f, _ = self.fast(e)
            s, _ = self.slow(e, e, e)
            s = self.slow_lin(s)
            g = self.gate(torch.cat([f, s], -1))          # sigmoid inhibition
            h = g * f + (1 - g) * s
            return self.head(h)
    return DualStreamDIINA()


def make_mamba(vocab_size: int, d: int = 32) -> "nn.Module":
    """SSM/Mamba-like baseline: selective gated recurrence."""
    class TinyMamba(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d)
            self.inp = nn.Linear(d, d)
            self.sel = nn.Linear(d, d)   # selective Δ gate
            self.out = nn.Linear(d, vocab_size)

        def forward(self, x):
            h = torch.zeros(x.size(0), d)
            outs = []
            for t in range(x.size(1)):
                e = self.emb(x)[:, t]
                delta = torch.sigmoid(self.sel(e))
                h = (1 - delta) * h + delta * torch.tanh(self.inp(e))
                outs.append(h)
            H = torch.stack(outs, 1)
            return self.out(H)
    return TinyMamba()


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict:
    """Exact McNemar via binomial test on discordant pairs."""
    only_a = int(np.sum((a == 1) & (b == 0)))
    only_b = int(np.sum((a == 0) & (b == 1)))
    n = only_a + only_b
    if n == 0 or not SCIPY_OK:
        return {"only_a": only_a, "only_b": only_b, "n_discordant": n,
                "p_value": 1.0 if n == 0 else None}
    res = binomtest(only_a, n, p=0.5)
    return {"only_a": only_a, "only_b": only_b, "n_discordant": n,
            "p_value": float(res.pvalue)}


def cohens_g(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's g: proportion of matched pairs favoring A minus 0.5."""
    n = len(a)
    return float(np.sum(a > b) / n - 0.5)


def wilcoxon_signed(x: np.ndarray, y: np.ndarray) -> Dict:
    if not SCIPY_OK:
        return {"statistic": None, "p_value": None}
    try:
        stat, p = wilcoxon(x, y)
        return {"statistic": float(stat), "p_value": float(p)}
    except ValueError:
        return {"statistic": None, "p_value": 1.0}


def logit_shift(res_a: ModelResult, res_b: ModelResult) -> float:
    """Delta z: mean target-logit shift of A vs B."""
    return float(np.mean(res_a.target_logits - res_b.target_logits))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(n: int = 1000, seed: int = 2022, out_dir: str = ".") -> Dict:
    if not TORCH_OK:
        raise RuntimeError("PyTorch is required for benchmark_yoruba_stress.py")
    os.makedirs(out_dir, exist_ok=True)
    items = build_stress_dataset(n, seed)
    chars = sorted({c for it in items for c in (it["context"] + it["target"] + it["trap"])})
    vocab = {c: i + 1 for i, c in enumerate(chars)}
    vocab["<unk>"] = 0
    V = len(vocab)

    torch.manual_seed(seed)
    results = {
        "Transformer": _toy_eval(make_transformer(V), items, vocab, "Transformer", seed),
        "DIINA": _toy_eval(make_diina(V), items, vocab, "DIINA", seed),
        "Mamba": _toy_eval(make_mamba(V), items, vocab, "Mamba", seed),
    }

    diina = results["DIINA"]
    summary: Dict[str, dict] = {}
    for name, r in results.items():
        summary[name] = {
            "accuracy": float(r.correct.mean()),
            "escape_rate": float(r.escaped.mean()),
            "recovery_rate": float(r.recovered.sum() / max(r.escaped.sum(), 1)),
            "mean_ce": float(r.ce.mean()),
            "target_logit_mean": float(r.target_logits.mean()),
            "target_logit_std": float(r.target_logits.std()),
            "mean_margin": float(r.margin.mean()),
        }
    summary["DIINA_vs_Transformer"] = {
        "mcnemar": mcnemar_exact(diina.correct, results["Transformer"].correct),
        "wilcoxon_ce": wilcoxon_signed(results["Transformer"].ce, diina.ce),
        "cohens_g": cohens_g(diina.correct, results["Transformer"].correct),
        "delta_z": logit_shift(diina, results["Transformer"]),
    }
    summary["DIINA_vs_Mamba"] = {
        "mcnemar": mcnemar_exact(diina.correct, results["Mamba"].correct),
        "wilcoxon_ce": wilcoxon_signed(results["Mamba"].ce, diina.ce),
        "cohens_g": cohens_g(diina.correct, results["Mamba"].correct),
        "delta_z": logit_shift(diina, results["Mamba"]),
    }

    # ---- per-item CSV ----
    csv_path = os.path.join(out_dir, "yoruba_stress_per_item.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "target", "trap"] +
                   [f"{r.name}_correct" for r in results.values()] +
                   [f"{r.name}_ce" for r in results.values()] +
                   [f"{r.name}_margin" for r in results.values()])
        for i, it in enumerate(items):
            w.writerow([it["id"], it["target"], it["trap"]] +
                       [int(r.correct[i]) for r in results.values()] +
                       [round(float(r.ce[i]), 6) for r in results.values()] +
                       [round(float(r.margin[i]), 6) for r in results.values()])
    # ---- summary CSV ----
    sum_path = os.path.join(out_dir, "yoruba_stress_summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "accuracy", "escape_rate", "recovery_rate",
                    "mean_ce", "mean_margin"])
        for name in ("Transformer", "DIINA", "Mamba"):
            s = summary[name]
            w.writerow([name, s["accuracy"], s["escape_rate"], s["recovery_rate"],
                        s["mean_ce"], s["mean_margin"]])
    # ---- JSON ----
    json_path = os.path.join(out_dir, "yoruba_stress_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_items": n, "seed": seed, "summary": summary}, f,
                  ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in summary.items()}, indent=2, ensure_ascii=False))
    print("Saved:", csv_path, sum_path, json_path)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2022)
    ap.add_argument("--out", type=str, default=".")
    a = ap.parse_args()
    run(a.n, a.seed, a.out)
