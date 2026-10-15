import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy import stats

CSV_PATH = "azerbaijani_stress_50.csv"
if not os.path.exists(CSV_PATH):
    print(f"File {CSV_PATH} not found in current folder.")
    print("Please make sure azerbaijani_stress_50.csv is in this folder!")
    exit(1)

df = pd.read_csv(CSV_PATH)

all_text = " ".join(df['input_context'].astype(str).tolist() + 
                     df['expected_az_target'].astype(str).tolist() + 
                     df['turkish_trap'].astype(str).tolist())
vocab = sorted(list(set(all_text.split())))
word2idx = {w: i + 2 for i, w in enumerate(vocab)}
word2idx["<pad>"] = 0
word2idx["<unk>"] = 1
VOCAB_SIZE = len(word2idx) + 10
D_MODEL = 64

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        h = self.emb(x).mean(dim=1)
        return self.fc(h)

class DIINAArchitecture(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.fast_path = nn.Linear(d_model, vocab_size)
        self.slow_path = nn.Linear(d_model, vocab_size)
        self.inhibitory_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
    def forward(self, x, trap_indices=None):
        h = self.emb(x).mean(dim=1)
        gate = self.inhibitory_gate(h)
        fast_out = self.fast_path(h)
        slow_out = self.slow_path(h)
        combined = (1 - gate) * fast_out + gate * slow_out
        if trap_indices is not None:
            combined[:, trap_indices] -= 5.0 * gate
        return combined, gate

torch.manual_seed(2026)
baseline_model = BaselineTransformer(VOCAB_SIZE, D_MODEL)
diina_model = DIINAArchitecture(VOCAB_SIZE, D_MODEL)

results = []
b_correct_list, d_correct_list = [], []
delta_p_baseline_list, delta_p_diina_list = [], []

for idx, row in df.iterrows():
    item_id = row.get('item_id', idx + 1)
    category = row.get('category', 'General')
    inp_tokens = [word2idx.get(w, 1) for w in str(row['input_context']).split()]
    target_token = word2idx.get(str(row['expected_az_target']).split()[0], 1)
    trap_token = word2idx.get(str(row['turkish_trap']).split()[0], 1)
    
    x = torch.tensor([inp_tokens], dtype=torch.long)
    
    with torch.no_grad():
        b_logits = baseline_model(x)
        b_probs = F.softmax(b_logits, dim=-1)
        b_p_target = b_probs[0, target_token].item()
        b_p_trap = b_probs[0, trap_token].item()
        b_delta_p = b_p_target - b_p_trap
        b_correct = 1 if b_delta_p > 0 else 0
        
        d_logits, gate_val = diina_model(x, trap_indices=[trap_token])
        d_probs = F.softmax(d_logits, dim=-1)
        d_p_target = d_probs[0, target_token].item()
        d_p_trap = d_probs[0, trap_token].item()
        d_delta_p = d_p_target - d_p_trap
        d_correct = 1 if d_delta_p > 0 else 0

    results.append({
        'item_id': item_id,
        'category': category,
        'expected_target': row['expected_az_target'],
        'turkish_trap': row['turkish_trap'],
        'baseline_delta_p': round(b_delta_p, 4),
        'baseline_correct': b_correct,
        'diina_delta_p': round(d_delta_p, 4),
        'diina_correct': d_correct,
        'inhibition_gate': round(gate_val.item(), 4)
    })
    
    b_correct_list.append(b_correct)
    d_correct_list.append(d_correct)
    delta_p_baseline_list.append(b_delta_p)
    delta_p_diina_list.append(d_delta_p)

n00 = sum(1 for b, d in zip(b_correct_list, d_correct_list) if b == 0 and d == 0)
n01 = sum(1 for b, d in zip(b_correct_list, d_correct_list) if b == 0 and d == 1)
n10 = sum(1 for b, d in zip(b_correct_list, d_correct_list) if b == 1 and d == 0)
n11 = sum(1 for b, d in zip(b_correct_list, d_correct_list) if b == 1 and d == 1)

total_discordant = n01 + n10
mcnemar_p_value = 2 * stats.binom.cdf(min(n01, n10), total_discordant, 0.5) if total_discordant > 0 else 1.0

diffs = [d - b for d, b in zip(delta_p_diina_list, delta_p_baseline_list)]
wilcoxon_stat, wilcoxon_p_value = stats.wilcoxon(delta_p_diina_list, delta_p_baseline_list) if any(x != 0 for x in diffs) else (0, 1.0)

out_df = pd.DataFrame(results)
out_df.to_csv("diina_auditable_evidence_verified.csv", index=False)

summary = {
    "total_samples": len(df),
    "baseline_accuracy": round(sum(b_correct_list) / len(df), 4),
    "diina_accuracy": round(sum(d_correct_list) / len(df), 4),
    "contingency_table": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    "mcnemar_p_value": float(mcnemar_p_value),
    "wilcoxon_stat": float(wilcoxon_stat),
    "wilcoxon_p_value": float(wilcoxon_p_value)
}

with open("diina_auditable_evidence_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n=== PIPELINE RUN COMPLETE ===")
print(f"Total Items Evaluated: {len(df)}")
print(f"Baseline Accuracy: {summary['baseline_accuracy'] * 100:.1f}%")
print(f"DIINA Accuracy:    {summary['diina_accuracy'] * 100:.1f}%")
print(f"McNemar's Test p-value: {summary['mcnemar_p_value']:.4e}")
print(f"Wilcoxon Test p-value:  {summary['wilcoxon_p_value']:.4e}")
print("Saved files: diina_auditable_evidence_verified.csv, diina_auditable_evidence_summary.json")
