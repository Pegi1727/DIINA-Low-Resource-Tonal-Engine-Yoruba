#!/usr/bin/env python3
"""Generate the Yoruba tonal benchmark CSV and publication-ready figure."""
import csv
import os
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/data"
CSV_PATH = os.path.join(OUT_DIR, "yoruba_tonal_benchmark_results.csv")
FIG_PATH = os.path.join(OUT_DIR, "figure_11_yoruba_tonal_benchmark.png")

rows = [
    ["DIINA", "90.67", "18.05", "7.10", "0.9999", "-0.7313"],
    ["Transformer", "91.00", "128.16", "1.00", "NaN", "NaN"],
]
header = ["Model", "Accuracy_Percent", "Training_Time_Sec", "Speedup_Factor",
          "Native_Alignment_Score", "Distorted_Penalty_Score"]

os.makedirs(OUT_DIR, exist_ok=True)
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), constrained_layout=True)
fig.suptitle("Yoruba Tonal Benchmark", fontsize=17, fontweight="bold", y=1.03)

# (a) Latency comparison
names = ["Transformer", "DIINA"]
latencies = [128.16, 18.05]
colors = ["#9AA6B2", "#247BA0"]
bars = ax1.bar(names, latencies, color=colors, width=0.58, edgecolor="white", linewidth=1.0)
ax1.set_title("(a) Training time / latency")
ax1.set_ylabel("Time (seconds)")
ax1.set_ylim(0, 155)
ax1.grid(axis="y", linestyle="--", alpha=0.35)
ax1.set_axisbelow(True)
for bar, val in zip(bars, latencies):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 3, f"{val:.2f}s", ha="center", va="bottom", fontweight="bold")
ax1.annotate("7.10x Speedup", xy=(1, 18.05), xytext=(0.58, 83),
             textcoords="data", ha="center", va="center", color="#145374",
             fontweight="bold", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.45", fc="#EAF5F8", ec="#247BA0", lw=1.4),
             arrowprops=dict(arrowstyle="->", color="#247BA0", lw=1.5,
                             connectionstyle="arc3,rad=0.15"))

# (b) Tonal pronunciation assessment scores
ax2.axhline(0, color="#333333", lw=1.0)
labels = ["Native vs\nCorrect", "Distorted /\nTonal Error"]
values = [0.9999, -0.7313]
bar_colors = ["#2A9D8F", "#E76F51"]
xs = [0, 1]
ax2.bar(xs, values, color=bar_colors, width=0.56, edgecolor="white", linewidth=1.0)
ax2.set_xticks(xs, labels)
ax2.set_ylabel("Pronunciation assessment score")
ax2.set_title("(b) Tonal pronunciation assessment")
ax2.set_ylim(-1.0, 1.25)
ax2.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
ax2.grid(axis="y", linestyle="--", alpha=0.35)
ax2.set_axisbelow(True)
ax2.text(0, 1.045, "+0.9999", ha="center", va="bottom", color="#1F776D", fontweight="bold", fontsize=12)
ax2.text(1, -0.78, "−0.7313", ha="center", va="top", color="#B94E37", fontweight="bold", fontsize=12)

for ax in (ax1, ax2):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"CSV: {CSV_PATH}")
print(f"Figure: {FIG_PATH}")
