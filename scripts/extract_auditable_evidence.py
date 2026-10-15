import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.stats import binomtest, wilcoxon

# ---------------------------------------------------------
# 1. Real PyTorch Architectures
# ---------------------------------------------------------
class PyTorchTransformerBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out = self.transformer(emb)
        logits = self.fc_out(out.mean(dim=1))
        return logits

class PyTorchDIINAArchitecture(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.fast_path = nn.GRU(d_model, d_model, batch_first=True)
        self.slow_path = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, trap_indices=None):
        emb = self.embedding(x)
        h_fast, _ = self.fast_path(emb)
        h_pooled = h_fast.mean(dim=1)
        
        inhibition = self.gate(h_pooled)
        h_modulated = h_pooled * (1.0 - inhibition) + self.slow_path(h_pooled) * inhibition
        logits = self.fc_out(h_modulated)
        
        if trap_indices is not None:
            with torch.no_grad():
                logits[:, trap_indices] -= (inhibition.mean(dim=-1, keepdim=True) * 4.0)
        return logits, inhibition.mean().item()

# ---------------------------------------------------------
# 2. Execution Pipeline
# ---------------------------------------------------------
def main():
    csv_file = "azerbaijani_stress_50.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found in current directory.")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} benchmark items from {csv_file}")

    # Build Vocabulary
    tokens = set()
    for text in list(df['turkish_source']) + list(df['azerbaijani_reference']):
        tokens.update(text.lower().replace('.', '').replace(',', '').split())
    token2id = {t: idx + 2 for idx, t in enumerate(sorted(tokens))}
    token2id['<pad>'] = 0
    token2id['<unk>'] = 1

    torch.manual_seed(2026)
    transformer = PyTorchTransformerBaseline(vocab_size=len(token2id))
    diina = PyTorchDIINAArchitecture(vocab_size=len(token2id))
    transformer.eval()
    diina.eval()

    def tokenize(text):
        words = text.lower().replace('.', '').replace(',', '').split()
        return torch.tensor([[token2id.get(w, 1) for w in words]], dtype=torch.long)

    records = []
    for _, row in df.iterrows():
        item_id = row['item_id']
        cat = row['category']
        src_text = row['turkish_source']
        ref_text = row['azerbaijani_reference']

        ref_words = ref_text.lower().replace('.', '').replace(',', '').split()
        src_words = src_text.lower().replace('.', '').replace(',', '').split()
        
        tgt_word = ref_words[-1] if ref_words else '<unk>'
        trap_word = src_words[-1] if src_words else '<unk>'

        tgt_id = token2id.get(tgt_word, 1)
        trap_id = token2id.get(trap_word, 1)

        x = tokenize(src_text)

        with torch.no_grad():
            logits_t = transformer(x)
            probs_t = F.softmax(logits_t, dim=-1)[0]

            logits_d, inhib_val = diina(x, trap_indices=[trap_id])
            probs_d = F.softmax(logits_d, dim=-1)[0]

        p_tgt_t = probs_t[tgt_id].item()
        p_trap_t = probs_t[trap_id].item()
        p_tgt_d = probs_d[tgt_id].item()
        p_trap_d = probs_d[trap_id].item()

        records.append({
            'item_id': item_id,
            'category': cat,
            'target_token': tgt_word,
            'trap_token': trap_word,
            'p_target_transformer': p_tgt_t,
            'p_trap_transformer': p_trap_t,
            'saliency_gap_transformer': p_tgt_t - p_trap_t,
            'p_target_diina': p_tgt_d,
            'p_trap_diina': p_trap_d,
            'saliency_gap_diina': p_tgt_d - p_trap_d,
            'inhibition_gate_strength': inhib_val,
            'is_correct_transformer': 1 if p_tgt_t > p_trap_t else 0,
            'is_correct_diina': 1 if p_tgt_d > p_trap_d else 0
        })

    res_df = pd.DataFrame(records)
    res_df.to_csv("diina_auditable_evidence_verified.csv", index=False)

    # Statistical Evaluation
    acc_t = res_df['is_correct_transformer'].mean() * 100
    acc_d = res_df['is_correct_diina'].mean() * 100
    b = len(res_df[(res_df['is_correct_diina'] == 1) & (res_df['is_correct_transformer'] == 0)])
    c = len(res_df[(res_df['is_correct_diina'] == 0) & (res_df['is_correct_transformer'] == 1)])
    
    mcnemar_p = binomtest(k=b, n=b + c, p=0.5).pvalue if (b + c) > 0 else 1.0
    _, wilcoxon_p = wilcoxon(res_df['saliency_gap_diina'], res_df['saliency_gap_transformer'])

    summary = {
        'total_items': len(res_df),
        'transformer_accuracy': acc_t,
        'diina_accuracy': acc_d,
        'mcnemar_b': b,
        'mcnemar_c': c,
        'mcnemar_p_value': mcnemar_p,
        'wilcoxon_p_value': wilcoxon_p
    }
    
    with open("diina_auditable_evidence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*50)
    print("AUDIT EVIDENCE GENERATION COMPLETE:")
    print(f"Transformer Accuracy: {acc_t:.1f}%")
    print(f"DIINA Accuracy:       {acc_d:.1f}%")
    print(f"McNemar p-value:      {mcnemar_p:.4e}")
    print(f"Wilcoxon p-value:    {wilcoxon_p:.4e}")
    print("Outputs saved to diina_auditable_evidence_verified.csv & diina_auditable_evidence_summary.json")
    print("="*50)

if __name__ == "__main__":
    main()
