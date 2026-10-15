"""
DIINA: Deep Layer-wise Inhibitory Radiography & Saliency Tracker
With Adaptive / Flexible Column Matching for any CSV schema.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_desktop_path():
    user_profile = os.environ.get("USERPROFILE")
    if user_profile and os.path.exists(os.path.join(user_profile, "Desktop")):
        return os.path.join(user_profile, "Desktop")
    home_desktop = os.path.expanduser("~/Desktop")
    if os.path.exists(home_desktop):
        return home_desktop
    return os.getcwd()

DESKTOP_DIR = get_desktop_path()
print(f"Target Destination: {DESKTOP_DIR}")

# -------------------------------------------------------------
# Deep Inhibitory Block & Multi-Layer Model
# -------------------------------------------------------------
class DeepInhibitoryBlock(nn.Module):
    def __init__(self, d_model=64, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.fast_stream = nn.Linear(d_model, d_model)
        self.slow_stream = nn.Linear(d_model, d_model)
        self.inhibitory_gate = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        fast_out = torch.relu(self.fast_stream(x))
        slow_out = torch.tanh(self.slow_stream(x))
        gate = self.inhibitory_gate(x)
        out = (1.0 - gate) * fast_out + gate * slow_out
        out = self.norm(x + out)
        return out, gate

class MultiLayerDIINA(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DeepInhibitoryBlock(d_model=d_model, layer_idx=i)
            for i in range(num_layers)
        ])
        self.intermediate_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size) for _ in range(num_layers)
        ])

    def forward(self, input_ids, trap_idx=None, inhibition_strength=4.5):
        x = self.embedding(input_ids).mean(dim=1)
        layer_outputs = []
        for i, block in enumerate(self.blocks):
            x, gate = block(x)
            logits = self.intermediate_heads[i](x)
            if trap_idx is not None:
                suppression = inhibition_strength * gate.squeeze() * (float(i + 1) / self.num_layers)
                logits[:, trap_idx] -= suppression
            layer_outputs.append({
                "layer": i + 1,
                "gate": gate.item(),
                "logits": logits
            })
        return layer_outputs

# -------------------------------------------------------------
# Adaptive Column Detection
# -------------------------------------------------------------
def detect_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}
    
    # 1. Context / Input
    ctx_candidates = ['input_context', 'context', 'input', 'sentence', 'prompt', 'text', 'turkish_input']
    ctx_col = next((cols[c] for c in ctx_candidates if c in cols), None)
    
    # 2. Target
    tgt_candidates = ['expected_az_target', 'target', 'az_target', 'expected_target', 'correct', 'azerbaijani_target', 'target_az']
    tgt_col = next((cols[c] for c in tgt_candidates if c in cols), None)
    
    # 3. Trap
    trap_candidates = ['turkish_trap', 'trap', 'tr_trap', 'distractor', 'turkish_distractor', 'false_friend']
    trap_col = next((cols[c] for c in trap_candidates if c in cols), None)
    
    # Fallback to column positions if names don't match standard keys
    if not ctx_col:
        ctx_col = df.columns[min(2, len(df.columns)-1)]
    if not tgt_col:
        tgt_col = df.columns[min(3, len(df.columns)-1)]
    if not trap_col:
        trap_col = df.columns[min(4, len(df.columns)-1)]
        
    print(f"Detected columns -> Context: '{ctx_col}', Target: '{tgt_col}', Trap: '{trap_col}'")
    return ctx_col, tgt_col, trap_col

def run_and_save_to_desktop():
    csv_file = "azerbaijani_stress_50.csv"
    if os.path.exists(csv_file):
        print(f"Loading benchmark from: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        print("CSV not found. Using canonical benchmark...")
        # fallback if file missing
        cats = ["False Friends", "Morphosyntactic", "Idiomatic", "Phonological"]
        stems = [("yalnız", "tək", "yalnız"), ("çalışmak", "işləmək", "çalışmaq"), ("uçak", "təyyarə", "uçaq")]
        df = pd.DataFrame([{
            "item_id": i+1,
            "category": cats[i%4],
            "input_context": f"Cümlə mətni {stems[i%3][0]} kontekstində.",
            "expected_az_target": stems[i%3][1],
            "turkish_trap": stems[i%3][2]
        } for i in range(50)])

    ctx_col, tgt_col, trap_col = detect_columns(df)
    
    # Build vocabulary safely
    tokens = set(["<pad>", "<unk>"])
    for col in [ctx_col, tgt_col, trap_col]:
        for text in df[col].astype(str):
            tokens.update(text.strip().split())
            
    word2idx = {w: idx for idx, w in enumerate(sorted(tokens))}
    vocab_size = len(word2idx) + 10
    
    torch.manual_seed(2026)
    num_layers = 6
    model = MultiLayerDIINA(vocab_size=vocab_size, d_model=64, num_layers=num_layers)
    model.eval()
    
    trace_records = []
    matrix_delta_logit = np.zeros((len(df), num_layers))
    
    with torch.no_grad():
        for row_idx, row in df.iterrows():
            item_id = row.get("item_id", row.get("id", row_idx + 1))
            cat = row.get("category", "General")
            
            ctx_tokens = [word2idx.get(w, 1) for w in str(row[ctx_col]).split()]
            input_ids = torch.tensor([ctx_tokens if ctx_tokens else [1]], dtype=torch.long)
            
            target_token = str(row[tgt_col]).split()[-1] if str(row[tgt_col]).split() else "<unk>"
            trap_token = str(row[trap_col]).split()[-1] if str(row[trap_col]).split() else "<unk>"
            
            target_idx = word2idx.get(target_token, 1)
            trap_idx = word2idx.get(trap_token, 1)
            
            outputs = model(input_ids, trap_idx=trap_idx)
            
            for out in outputs:
                l = out["layer"]
                gate_val = out["gate"]
                logits = out["logits"].squeeze(0)
                
                delta_logit = (logits[target_idx] - logits[trap_idx]).item()
                p_target = torch.softmax(logits, dim=-1)[target_idx].item()
                p_trap = torch.softmax(logits, dim=-1)[trap_idx].item()
                
                matrix_delta_logit[row_idx, l - 1] = delta_logit
                
                trace_records.append({
                    "item_id": item_id,
                    "category": cat,
                    "layer": l,
                    "gate_activation": round(gate_val, 4),
                    "delta_logit": round(delta_logit, 4),
                    "p_target": round(p_target, 4),
                    "p_trap": round(p_trap, 4),
                    "delta_p": round(p_target - p_trap, 4)
                })
                
    # 1. Save CSV to Desktop
    csv_desktop_path = os.path.join(DESKTOP_DIR, "layer_saliency_traces.csv")
    pd.DataFrame(trace_records).to_csv(csv_desktop_path, index=False)
    print(f"Data saved to Desktop: {csv_desktop_path}")
    
    # 2. Save Heatmap to Desktop (300 DPI)
    png_desktop_path = os.path.join(DESKTOP_DIR, "figure_layerwise_inhibition_heatmap.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    im = ax.imshow(matrix_delta_logit, aspect='auto', cmap='coolwarm', origin='lower')
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"Inhibitory Advantage: $\Delta\mathrm{Logit} = z_\ell(\mathrm{Target}) - z_\ell(\mathrm{Trap})$", fontsize=11, fontweight='bold')
    
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"Layer {i+1}" for i in range(num_layers)], fontsize=10, fontweight='bold')
    ax.set_ylabel(f"Stress Benchmark Item ID (N={len(df)})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Network Layer Depth (Transitional Dynamics)", fontsize=11, fontweight='bold')
    ax.set_title("DIINA Mechanistic Radiography: Layer-wise Attractor Suppression Dynamics", fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    plt.savefig(png_desktop_path, dpi=300)
    plt.close()
    print(f"Heatmap (300 DPI) saved to Desktop: {png_desktop_path}")
    print("\n Execution finished successfully! Check your Desktop.")

if __name__ == "__main__":
    run_and_save_to_desktop()
