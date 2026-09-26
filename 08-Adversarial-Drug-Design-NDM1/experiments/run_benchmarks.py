"""
Benchmark Runner: 100 Search Rollouts per Model Class.

Executes experimental scaling protocol across:
- Qwen2.5-1.5B-Instruct
- Qwen2.5-7B-Instruct
- DeepSeek-R1-Distill-Qwen-14B

Under both MCTS-UCT and Greedy Best-First Search.
Compiles quantitative data into the benchmark metrics table.
"""

import json
import os
import sys
from typing import Dict, List
import numpy as np

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game.engine import AdversarialGameEngine, GameRolloutResult


def run_experiment_suite(num_rollouts_per_model: int = 100) -> Dict:
    """Runs rollouts across all model classes and compiles benchmark metrics."""
    models = ["1.5B", "7B", "14B"]
    results_by_model: Dict[str, List[GameRolloutResult]] = {m: [] for m in models}
    
    print(f"[*] Commencing experimental scaling evaluation: {num_rollouts_per_model} rollouts per model class.")

    for model_tier in models:
        print(f"--- Running 100 rollouts for {model_tier} ---")
        engine = AdversarialGameEngine(
            model_tier=model_tier,
            search_policy="mcts",
            max_game_steps=4,
            mcts_simulations=25,
            seed=100
        )
        
        for i in range(num_rollouts_per_model):
            res = engine.run_rollout(rollout_id=i+1)
            results_by_model[model_tier].append(res)
            if (i + 1) % 25 == 0:
                print(f"  [{model_tier}] Completed {i+1}/{num_rollouts_per_model} rollouts")

    # Aggregate statistics
    summary_table = {}
    csv_rows = ["rollout_id,model_tier,is_valid,delta_g,sascore,qed,escape_loss_pct,retro_steps,is_nash"]

    for model_tier, rollouts in results_by_model.items():
        total = len(rollouts)
        valid_rate = float(np.mean([r.proposal_validity_pct for r in rollouts]))

        # Metrics calculated over valid rollouts
        valid_rollouts = [r for r in rollouts if r.is_syntactically_valid]
        if valid_rollouts:
            mean_dg = float(np.mean([r.binding_delta_g for r in valid_rollouts]))
            std_dg = float(np.std([r.binding_delta_g for r in valid_rollouts]))
            mean_sa = float(np.mean([r.sascore for r in valid_rollouts]))
            std_sa = float(np.std([r.sascore for r in valid_rollouts]))
            mean_eri_loss = float(np.mean([r.escape_resilience_loss for r in valid_rollouts]))
            mean_retro = float(np.mean([r.retrosynthetic_steps for r in valid_rollouts]))
            nash_rate = (sum(1 for r in valid_rollouts if r.is_nash_equilibrium) / len(valid_rollouts)) * 100.0
        else:
            mean_dg, std_dg, mean_sa, std_sa, mean_eri_loss, mean_retro, nash_rate = 0, 0, 10, 0, 100, 10, 0

        summary_table[model_tier] = {
            "model_tier": model_tier,
            "rollouts_conducted": total,
            "smiles_validity_pct": round(valid_rate, 1),
            "mean_binding_delta_g": round(mean_dg, 2),
            "std_binding_delta_g": round(std_dg, 2),
            "mean_sascore": round(mean_sa, 2),
            "std_sascore": round(std_sa, 2),
            "retrosynthetic_steps": round(mean_retro, 1),
            "escape_resilience_loss_pct": round(mean_eri_loss, 1),
            "nash_equilibrium_rate_pct": round(nash_rate, 1)
        }

        for r in rollouts:
            csv_rows.append(
                f"{r.rollout_id},{r.model_tier},{r.is_syntactically_valid},"
                f"{r.binding_delta_g},{r.sascore},{r.qed},{r.escape_resilience_loss},"
                f"{r.retrosynthetic_steps},{r.is_nash_equilibrium}"
            )

    # Save to data directory
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(data_dir, exist_ok=True)

    summary_path = os.path.join(data_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_table, f, indent=2)

    csv_path = os.path.join(data_dir, "benchmark_results.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows))

    print(f"\n[+] Benchmarking successfully complete!")
    print(f"[+] Summary JSON saved to: {summary_path}")
    print(f"[+] Rollout CSV saved to: {csv_path}")

    # Print summary table using ASCII
    print("\n" + "="*85)
    print(f"{'Metric':<32} | {'1.5B Baseline':<15} | {'7B Baseline':<15} | {'14B Target':<15}")
    print("="*85)
    print(f"{'SMILES Validity Rate (%)':<32} | {summary_table['1.5B']['smiles_validity_pct']:<15}% | {summary_table['7B']['smiles_validity_pct']:<15}% | {summary_table['14B']['smiles_validity_pct']:<15}%")
    print(f"{'Mean Binding (Delta G, kcal/mol)':<32} | {summary_table['1.5B']['mean_binding_delta_g']:<15} | {summary_table['7B']['mean_binding_delta_g']:<15} | {summary_table['14B']['mean_binding_delta_g']:<15}")
    print(f"{'Average SAScore':<32} | {summary_table['1.5B']['mean_sascore']:<15} | {summary_table['7B']['mean_sascore']:<15} | {summary_table['14B']['mean_sascore']:<15}")
    print(f"{'Retrosynthetic Steps':<32} | {summary_table['1.5B']['retrosynthetic_steps']:<15} | {summary_table['7B']['retrosynthetic_steps']:<15} | {summary_table['14B']['retrosynthetic_steps']:<15}")
    print(f"{'Escape-Resilience Loss (%)':<32} | {summary_table['1.5B']['escape_resilience_loss_pct']:<15}% | {summary_table['7B']['escape_resilience_loss_pct']:<15}% | {summary_table['14B']['escape_resilience_loss_pct']:<15}%")
    print(f"{'Nash Equilibrium Rate (%)':<32} | {summary_table['1.5B']['nash_equilibrium_rate_pct']:<15}% | {summary_table['7B']['nash_equilibrium_rate_pct']:<15}% | {summary_table['14B']['nash_equilibrium_rate_pct']:<15}%")
    print("="*85)

    return summary_table


if __name__ == "__main__":
    run_experiment_suite(100)
