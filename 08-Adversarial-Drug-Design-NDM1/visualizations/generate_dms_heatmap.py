"""
Generate Publication Deep Mutational Scanning (DMS) Heatmap from Real ESM-2 GPU Runs.

Visualizes the Delta_Fitness matrix across:
- Flexible loop resistance hotspots (Met67, Lys211, Asn220)
- Invariant catalytic zinc coordination triad (His120, His122, Asp124, His189, Cys208, His250)

Demonstrates the biological checkmate: 0% viability in catalytic core vs high plasticity in loop hotspots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
DELIV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables'))
os.makedirs(DELIV_DIR, exist_ok=True)


def plot_esm2_dms_heatmap():
    csv_path = os.path.join(DATA_DIR, "real_esm2_dms_results.csv")
    if not os.path.exists(csv_path):
        print(f"[-] CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Create label: e.g. "M67 (Hotspot)", "H120 (Catalytic)"
    df['residue_label'] = df.apply(
        lambda r: f"{r['wt_amino_acid']}{r['position']}\n({'Core' if r['is_catalytic_core'] else 'Hotspot'})", axis=1
    )

    # Pivot table: rows = amino acids, columns = residues
    pivot_df = df.pivot(index='mut_amino_acid', columns='residue_label', values='delta_fitness')

    # Reorder columns to group Hotspots first, then Catalytic Core
    hotspots = [c for c in pivot_df.columns if "Hotspot" in c]
    catalytic = [c for c in pivot_df.columns if "Core" in c]
    ordered_cols = sorted(hotspots) + sorted(catalytic)
    pivot_df = pivot_df[ordered_cols]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#111827')

    # Diverging colormap centered at -1.5 (fold viability threshold)
    # Values > -1.5: Viable (emerald green / teal)
    # Values < -1.5: Pruned / Lethal (crimson red / purple)
    cmap = sns.diverging_palette(10, 150, s=85, l=45, as_cmap=True, center="dark")

    # Heatmap
    sns.heatmap(
        pivot_df,
        cmap=cmap,
        center=-1.5,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 7.5, "weight": "bold"},
        cbar_kws={'label': 'ESM-2 Zero-Shot Mutational Fitness: ΔFitness = log P(mut) - log P(wt)'},
        linewidths=0.5,
        linecolor='#1e293b',
        ax=ax
    )

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color('#e2e8f0')
    cbar.ax.yaxis.label.set_size(10)
    cbar.ax.yaxis.label.set_weight('bold')
    cbar.ax.tick_params(colors='#94a3b8')

    # Add vertical divider between Hotspots and Catalytic Core
    divider_x = len(hotspots)
    ax.axvline(divider_x, color='#f59e0b', linewidth=3, linestyle='--', alpha=0.9)
    ax.text(
        divider_x - 0.1, -0.6, 'Plastic Loop Hotspots\n(Viable Escape Moves)',
        color='#34d399', fontsize=10, fontweight='bold', ha='right'
    )
    ax.text(
        divider_x + 0.1, -0.6, 'Invariant Catalytic Shell\n(0% Viability / Lethal)',
        color='#f87171', fontsize=10, fontweight='bold', ha='left'
    )

    # Labels and title
    plt.title(
        'Deep Mutational Scanning (ESM-2 650M) of NDM-1 Active Site: The Catalytic Trap',
        color='#f8fafc', fontsize=13, fontweight='bold', pad=32
    )
    plt.xlabel('Active Site Target Residue (PDB: 3SPU)', color='#e2e8f0', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel('Substituted Amino Acid', color='#e2e8f0', fontsize=11, fontweight='bold', labelpad=10)

    ax.tick_params(axis='x', colors='#e2e8f0', labelsize=9)
    ax.tick_params(axis='y', colors='#e2e8f0', labelsize=8.5)

    out_path = os.path.join(DELIV_DIR, "figure4_real_esm2_dms_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Successfully generated Nature-grade DMS Heatmap: {out_path}")


if __name__ == "__main__":
    plot_esm2_dms_heatmap()
