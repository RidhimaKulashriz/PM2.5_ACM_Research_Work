from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DML = ROOT / 'data/modeling_changes/dml_v3'
FIG = DML / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'figure.dpi': 160, 'savefig.dpi': 220, 'font.size': 9, 'axes.titleweight': 'bold'})

base = pd.read_csv(DML / 'dml_summary.csv')
robust = pd.read_csv(DML / 'robustness_summary.csv')
time = pd.read_csv(DML / 'pre_treatment_dml_summary.csv')
labels = {
    'sentinel2_ndvi_mean_1000m': 'Sentinel-2 NDVI 1,000 m',
    'sentinel2_ndvi_mean_500m': 'Sentinel-2 NDVI 500 m',
    'modis_ndvi_mean_1000m': 'MODIS NDVI 1,000 m',
}
order = list(labels)

# Figure 1: base and dependence-aware intervals.
fig, ax = plt.subplots(figsize=(8.2, 4.3))
y = np.arange(len(order))
for i, treatment in enumerate(order):
    b = base.loc[base.treatment == treatment].iloc[0]
    r = robust.loc[robust.treatment == treatment].iloc[0]
    ax.plot([b.ci_low, b.ci_high], [i - 0.11, i - 0.11], color='#4c78a8', lw=5, solid_capstyle='round', label='Original IF interval' if i == 0 else None)
    ax.scatter(b.theta, i - 0.11, color='#1f4e79', s=36, zorder=3)
    ax.plot([r.cluster_ci_low, r.cluster_ci_high], [i + 0.11, i + 0.11], color='#e07a5f', lw=5, solid_capstyle='round', label='Station-clustered interval' if i == 0 else None)
    ax.scatter(r.theta_robust, i + 0.11, color='#b23a48', s=36, zorder=3)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(y, [labels[x] for x in order])
ax.set_xlabel('PM₂.₅ µg/m³ per one-unit NDVI increase')
ax.set_title('V3 DML estimates: original versus dependence-aware uncertainty')
ax.legend(frameon=False, loc='lower right')
ax.grid(axis='x', alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / 'dml_estimates_forest.png', bbox_inches='tight')
plt.close(fig)

# Figure 2: time-aware lagged treatment estimates.
fig, ax = plt.subplots(figsize=(8.2, 4.3))
y = np.arange(len(order))
for i, treatment in enumerate(order):
    row = time.loc[time.treatment == treatment].iloc[0]
    ax.plot([row.time_aware_ci_low, row.time_aware_ci_high], [i, i], color='#59a14f', lw=6, solid_capstyle='round', label='Time-aware clustered interval' if i == 0 else None)
    ax.scatter(row.time_aware_theta, i, color='#2f6f3e', s=42, zorder=3)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(y, [labels[x] for x in order])
ax.set_xlabel('PM₂.₅ µg/m³ per one-unit lagged NDVI increase')
ax.set_title('Pre-treatment lagged NDVI: expanding time-aware DML')
ax.legend(frameon=False, loc='lower right')
ax.grid(axis='x', alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / 'time_aware_lagged_forest.png', bbox_inches='tight')
plt.close(fig)

# Figure 3: sample sizes by time holdout.
folds = pd.read_csv(DML / 'time_aware_dml_folds.csv')
fig, ax = plt.subplots(figsize=(8.2, 3.7))
for treatment in order:
    f = folds.loc[folds.treatment == treatment]
    ax.plot(f.holdout_year, f.n_holdout, marker='o', lw=2, label=labels[treatment])
ax.set_xticks(sorted(folds.holdout_year.unique()))
ax.set_ylabel('Rows in holdout year')
ax.set_xlabel('Holdout year')
ax.set_title('Expanding time-aware cross-fitting coverage')
ax.legend(frameon=False, ncol=1)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(FIG / 'time_aware_sample_coverage.png', bbox_inches='tight')
plt.close(fig)
print('FIGURES_GENERATED', sorted(p.name for p in FIG.glob('*.png')))
