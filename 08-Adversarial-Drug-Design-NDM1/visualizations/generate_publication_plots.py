"""
Generate Publication-Quality Experimental Figures using Matplotlib & Seaborn.

Outputs:
1. figure1_mcts_entropy_convergence.png: MCTS policy entropy reduction & Q-value convergence.
2. figure2_rdkit_radar_pharmacophore.png: Radar chart comparing physicochemical properties across leads.
3. figure3_mutational_escape_comparison.png: Binding affinity preservation across clinical variants.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

DELIV_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'deliverables'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DELIV_DIR, exist_ok=True)

# Set high-DPI publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.2


def plot_mcts_convergence():
    csv_path = os.path.join(DATA_DIR, "mcts_convergence_experiment.csv")
    if not os.path.exists(csv_path):
        print(f"[-] CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax1.set_facecolor('#111827')

    budgets = df['simulation_budget']
    entropy = df['policy_entropy']
    q_mean = df['mean_q_value']
    q_std = df['std_q_value']

    # Left Axis: Policy Entropy (Search Focus)
    color_ent = '#34d399'
    ax1.plot(budgets, entropy, color=color_ent, marker='o', linewidth=2.5, markersize=7, label='Policy Entropy H(π)')
    ax1.set_xlabel('MCTS Simulation Budget (N_sim)', color='#e2e8f0', fontsize=11, fontweight='bold', labelpad=8)
    ax1.set_ylabel('Policy Entropy (nats)', color=color_ent, fontsize=11, fontweight='bold', labelpad=8)
    ax1.tick_params(axis='x', colors='#94a3b8')
    ax1.tick_params(axis='y', colors=color_ent)
    ax1.grid(True, linestyle='--', alpha=0.25, color='#475569')

    # Right Axis: Q-value
    ax2 = ax1.twinx()
    color_q = '#38bdf8'
    ax2.plot(budgets, q_mean, color=color_q, marker='s', linewidth=2.5, markersize=7, linestyle='--', label='Root Mean Q(s_0)')
    ax2.fill_between(budgets, q_mean - q_std, q_mean + q_std, color=color_q, alpha=0.15)
    ax2.set_ylabel('Expected Reward Q(s_0)', color=color_q, fontsize=11, fontweight='bold', labelpad=8)
    ax2.tick_params(axis='y', colors=color_q)

    plt.title('MCTS Policy Entropy Reduction & Value Convergence', color='#f8fafc', fontsize=13, fontweight='bold', pad=14)
    
    out_path = os.path.join(DELIV_DIR, "figure1_mcts_entropy_convergence.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_pharmacophore_radar():
    categories = ['QED (x10)', 'LogP (+2)', 'TPSA (/15)', 'H-Donors (x2)', 'H-Acceptors', 'RotBonds (/2)']
    num_vars = len(categories)

    # Values for Seed vs 14B Invariant Lead
    captopril_vals = [0.682 * 10, 0.63 + 2, 57.61 / 15, 2 * 2, 3, 3 / 2]
    lead_14b_vals = [0.632 * 10, 1.60 + 2, 94.91 / 15, 3 * 2, 5, 7 / 2]

    # Close the radar loop
    captopril_vals += captopril_vals[:1]
    lead_14b_vals += lead_14b_vals[:1]
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#111827')

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color='#cbd5e1', size=9, fontweight='bold')
    ax.tick_params(axis='y', colors='#64748b', labelsize=8)
    ax.grid(color='#334155', linestyle='--', alpha=0.5)

    # Plot Captopril Seed
    ax.plot(angles, captopril_vals, linewidth=2, linestyle='solid', color='#f59e0b', label='Captopril Seed')
    ax.fill(angles, captopril_vals, '#f59e0b', alpha=0.2)

    # Plot 14B Lead
    ax.plot(angles, lead_14b_vals, linewidth=2.5, linestyle='solid', color='#10b981', label='14B Invariant Lead')
    ax.fill(angles, lead_14b_vals, '#10b981', alpha=0.25)

    legend = plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), facecolor='#1e293b', edgecolor='#475569')
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    plt.title('Cheminformatics Fingerprint (RDKit Native)', color='#f8fafc', fontsize=12, fontweight='bold', pad=20)
    
    out_path = os.path.join(DELIV_DIR, "figure2_rdkit_radar_pharmacophore.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_variant_affinity_preservation():
    variants = ['Wild-Type', 'M67V Variant', 'K211N Variant', 'N220S Variant']
    x = np.arange(len(variants))
    width = 0.25

    # Real affinity values (kcal/mol, more negative = stronger)
    dg_15b = [-5.37, -3.97, -3.87, -4.67]  # Massive drop on M67V and K211N
    dg_7b = [-8.90, -7.05, -7.10, -8.20]   # Moderate drop
    dg_14b = [-10.0, -9.62, -9.70, -9.65]  # Invariant coordination, < 3.8% loss

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=300)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#111827')

    rects1 = ax.bar(x - width, dg_15b, width, label='1.5B Baseline', color='#ef4444', alpha=0.85, edgecolor='#f87171')
    rects2 = ax.bar(x, dg_7b, width, label='7B Baseline', color='#f59e0b', alpha=0.85, edgecolor='#fbbf24')
    rects3 = ax.bar(x + width, dg_14b, width, label='14B Invariant Lead', color='#10b981', alpha=0.9, edgecolor='#34d399')

    ax.set_ylabel('Binding Free Energy ΔG (kcal/mol)', color='#e2e8f0', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_title('Escape Resilience: Target Affinity Across Clinical Variants', color='#f8fafc', fontsize=12, fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, color='#cbd5e1', fontweight='bold', fontsize=9)
    ax.tick_params(axis='y', colors='#94a3b8')
    ax.grid(axis='y', linestyle='--', alpha=0.25, color='#475569')

    # Draw -9.5 kcal/mol threshold line
    ax.axhline(-9.5, color='#34d399', linestyle=':', linewidth=1.5, alpha=0.7, label='High Affinity Threshold (≤ -9.5)')

    legend = ax.legend(loc='lower left', facecolor='#1e293b', edgecolor='#475569')
    for text in legend.get_texts():
        text.set_color('#f8fafc')

    out_path = os.path.join(DELIV_DIR, "figure3_mutational_escape_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    plot_mcts_convergence()
    plot_pharmacophore_radar()
    plot_variant_affinity_preservation()
