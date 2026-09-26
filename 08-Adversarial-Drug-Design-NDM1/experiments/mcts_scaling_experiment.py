"""
Experiment B: Monte Carlo Tree Search (MCTS) Scaling and Convergence Dynamics.

Evaluates how search simulation budget (N_sim = 5, 15, 30, 60, 120) impacts:
1. Root node value convergence Q(s_0)
2. Terminal Nash Equilibrium discovery rate (%)
3. Tree search depth reached
4. Policy entropy reduction: H(pi) = -sum p_i * ln(p_i)
5. Runtime latency (ms per move)
"""

import math
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game.engine import AdversarialGameEngine
from src.game.game_state import GameState
from src.pathogen.ndm1_target import PocketState
from src.chemist.scaffolds import SEED_SCAFFOLDS

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
os.makedirs(DATA_DIR, exist_ok=True)


def run_mcts_scaling_study(num_trials_per_budget: int = 15):
    print("[*] Starting Experiment B: MCTS Simulation Scaling & Value Convergence...\n")
    simulation_budgets = [5, 15, 30, 60, 100]
    records = []

    for budget in simulation_budgets:
        print(f"--- Testing Simulation Budget N_sim = {budget} ({num_trials_per_budget} trials) ---")
        q_values = []
        nash_hits = 0
        latencies = []
        steps_to_terminal = []
        entropies = []

        for trial in range(num_trials_per_budget):
            engine = AdversarialGameEngine(
                model_tier="14B",
                search_policy="mcts",
                max_game_steps=4,
                mcts_simulations=budget,
                seed=42 + trial
            )

            start_t = time.perf_counter()
            initial_state = GameState(
                pocket=PocketState(mutations={}),
                smiles=SEED_SCAFFOLDS["captopril_core"].smiles,
                turn=0,
                step_count=0,
                max_steps=4
            )

            # Perform root search
            best_child, children = engine.mcts.search(initial_state)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            latencies.append(elapsed_ms)

            # Compute visit distribution entropy
            visits = np.array([c.visits for c in children if c.visits > 0])
            if len(visits) > 1 and visits.sum() > 0:
                probs = visits / visits.sum()
                entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
            else:
                entropy = 0.0
            entropies.append(entropy)

            # Complete game rollout
            res = engine.run_rollout(rollout_id=trial + 1)
            q_values.append(best_child.q_value)
            steps_to_terminal.append(res.total_steps)
            if res.is_nash_equilibrium:
                nash_hits += 1

        mean_q = float(np.mean(q_values))
        std_q = float(np.std(q_values))
        nash_rate = (nash_hits / num_trials_per_budget) * 100.0
        mean_lat = float(np.mean(latencies))
        mean_ent = float(np.mean(entropies))
        mean_steps = float(np.mean(steps_to_terminal))

        print(f"[+] Budget N={budget:<3} | Q-value: {mean_q:.2f} +/- {std_q:.2f} | Nash Rate: {nash_rate:5.1f}% | Entropy: {mean_ent:.3f} | Latency: {mean_lat:6.1f} ms")

        records.append({
            "simulation_budget": budget,
            "mean_q_value": round(mean_q, 3),
            "std_q_value": round(std_q, 3),
            "nash_equilibrium_rate_pct": round(nash_rate, 1),
            "policy_entropy": round(mean_ent, 3),
            "mean_latency_ms": round(mean_lat, 2),
            "mean_game_steps": round(mean_steps, 2)
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(DATA_DIR, "mcts_convergence_experiment.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved MCTS scaling dataset to: {csv_path}\n")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_mcts_scaling_study(15)
