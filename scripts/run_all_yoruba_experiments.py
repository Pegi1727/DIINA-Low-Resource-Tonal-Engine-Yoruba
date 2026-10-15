#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all_yoruba_experiments.py
==============================================================================
Master orchestrator: runs the full Yoruba specialized-suite end-to-end:
  1. benchmark_yoruba_stress        (1000-item stress, stats, CSV/JSON)
  2. adversarial_diacritic_eval     (tonal-blindness robustness curves)
  3. train_yoruba_diina             (dual-stream training + minimal pairs)
  4. radiography_yoruba_layerwise   (gates + Delta-logit traces, heatmap)
  5. statistical_validation_suite   (bootstrap CI, McNemar, OR, Cohen's g, LaTeX)

All artifacts land in ./yoruba_results/ (configurable).
Run:  python run_all_yoruba_experiments.py [--steps all]
==============================================================================
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "yoruba_results")

STEPS = [
    ("benchmark",    "benchmark_yoruba_stress",        "run"),
    ("adversarial",  "adversarial_diacritic_eval",     "run"),
    ("train",        "train_yoruba_diina",             "main"),
    ("radiography",  "radiography_yoruba_layerwise",   "main"),
    ("stats",        "statistical_validation_suite",   "main"),
]


def run_step(name: str, module_name: str, entry: str) -> bool:
    print("=" * 70)
    print(f"[run_all] step: {name}")
    print("=" * 70)
    t0 = time.time()
    try:
        sys.path.insert(0, HERE)
        mod = importlib.import_module(module_name)
        fn = getattr(mod, entry)
        if name == "benchmark":
            fn(n=1000, seed=2022, out_dir=OUT_DIR)
        elif name == "adversarial":
            fn(n_items=1000, seed=2022, out=OUT_DIR)
        elif name == "train":
            # keep the orchestrator fast: short training on synthetic corpus
            import torch
            from train_yoruba_diina import (DualStreamDIINA, synthetic_corpus,
                                            build_vocab, train,
                                            eval_minimal_pairs)
            pairs = synthetic_corpus(500)
            stoi, _ = build_vocab(pairs)
            model = train(DualStreamDIINA(len(stoi)), pairs, stoi, epochs=3)
            eval_minimal_pairs(model, stoi, {i: c for c, i in stoi.items()})
            torch.save({"state_dict": model.state_dict(), "stoi": stoi},
                       os.path.join(OUT_DIR, "diina_yoruba.pt"))
        elif name == "radiography":
            fn(checkpoint=os.path.join(OUT_DIR, "diina_yoruba.pt"), out=OUT_DIR)
        elif name == "stats":
            res_json = os.path.join(OUT_DIR, "yoruba_stress_results.json")
            fn(results=res_json, out=OUT_DIR)
        print(f"[run_all] {name} OK in {time.time() - t0:.1f}s")
        return True
    except Exception:
        traceback.print_exc()
        print(f"[run_all] {name} FAILED")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, default="all",
                    help="comma list of: benchmark,adversarial,train,"
                         "radiography,stats  (or 'all')")
    ap.add_argument("--out", type=str, default=OUT_DIR)
    a = ap.parse_args()
    global OUT_DIR
    OUT_DIR = a.out
    os.makedirs(OUT_DIR, exist_ok=True)

    wanted = ([s[0] for s in STEPS] if a.steps == "all"
              else [s.strip() for s in a.steps.split(",")])
    status = {}
    for name, mod, entry in STEPS:
        if name in wanted:
            status[name] = run_step(name, mod, entry)

    manifest = os.path.join(OUT_DIR, "run_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"status": status, "output_dir": OUT_DIR}, f, indent=2)
    print("\n[run_all] summary:", json.dumps(status))
    print("[run_all] artifacts in", OUT_DIR)
    print("[run_all] manifest:", manifest)


if __name__ == "__main__":
    main()
