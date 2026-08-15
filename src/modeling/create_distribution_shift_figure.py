from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/figures/modeling/distribution_shift_by_month.png"
train = pd.read_csv(ROOT / "data/modeling/splits/train.csv")
test = pd.read_csv(ROOT / "data/modeling/splits/test.csv")

stats = []
for split, df in [("Train", train), ("Test", test)]:
    grouped = df.groupby("month", as_index=False).agg(mean_pm25=("pm25", "mean"), rows=("pm25", "size"))
    grouped["split"] = split
    stats.append(grouped)
stats = pd.concat(stats, ignore_index=True)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
colors = {"Train": "#2563eb", "Test": "#dc2626"}
for split in ["Train", "Test"]:
    part = stats[stats["split"] == split]
    axes[0].plot(part["month"], part["mean_pm25"], marker="o", linewidth=2.2, label=split, color=colors[split])
    axes[1].bar(part["month"] + (-0.18 if split == "Train" else 0.18), part["rows"], width=0.34, label=split, color=colors[split], alpha=0.85)

axes[0].set_ylabel("Mean PM$_{2.5}$")
axes[0].set_title("Locked split distribution shift by calendar month")
axes[0].grid(axis="y", alpha=0.25)
axes[0].legend(frameon=False, ncol=2)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Observations")
axes[1].set_xticks(range(1, 13))
axes[1].grid(axis="y", alpha=0.25)
axes[1].legend(frameon=False, ncol=2)
fig.savefig(OUT, dpi=220, bbox_inches="tight")
print(OUT)
