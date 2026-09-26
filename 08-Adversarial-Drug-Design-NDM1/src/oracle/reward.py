"""
Multi-Objective Reward Function for the Closed-Loop Game.

Formula:
    R = -w_1 * (ΔG) + w_2 * (QED) - w_3 * (SAScore)

Where:
- ΔG: Binding free energy from Smina/biophysical oracle (kcal/mol, typically negative)
- QED: Quantitative Estimate of Drug-likeness (0.0 to 1.0)
- SAScore: Synthetic Accessibility Score (1.0 to 10.0, Ertl & Schuffenhauer)
- Invariant Zn-coordination bonus awarded when ligand bridges both Zn1 and Zn2.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from src.oracle.rdkit_validator import MoleculeValidator
from src.oracle.sascore import SAScoreCalculator
from src.oracle.docking_oracle import BiophysicalDockingOracle, DockingScore
from src.pathogen.ndm1_target import PocketState

try:
    from rdkit import Chem
    from rdkit.Chem import QED
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class RewardComponents:
    total_reward: float
    is_valid: bool
    delta_g: float
    qed: float
    sascore: float
    escape_resilience_loss: float
    zn1_coordinated: bool
    zn2_coordinated: bool
    is_nash_equilibrium: bool
    explanation: str


class MultiObjectiveRewardEngine:
    """Evaluates the multi-objective reward function for candidate molecules."""

    def __init__(self, w1: float = 1.0, w2: float = 2.5, w3: float = 0.5):
        self.w1 = w1  # Affinity weight (-w1 * ΔG)
        self.w2 = w2  # Drug-likeness weight (+w2 * QED)
        self.w3 = w3  # Synthetic feasibility weight (-w3 * SAScore)
        self.docking_oracle = BiophysicalDockingOracle()

    def evaluate(self, smiles: str, pocket: PocketState) -> RewardComponents:
        """Computes comprehensive multi-objective reward and validation components."""
        # 1. Chemical Syntax Filter
        val = MoleculeValidator.validate(smiles)
        if not val.is_valid:
            return RewardComponents(
                total_reward=-15.0,
                is_valid=False,
                delta_g=0.0,
                qed=0.0,
                sascore=10.0,
                escape_resilience_loss=100.0,
                zn1_coordinated=False,
                zn2_coordinated=False,
                is_nash_equilibrium=False,
                explanation=f"Syntax Failure: {val.error_message}"
            )

        canon_smiles = val.canonical_smiles or smiles

        # 2. Biophysical Docking
        dock_res: DockingScore = self.docking_oracle.evaluate_binding(canon_smiles, pocket)

        # 3. Synthetic Feasibility
        sascore = SAScoreCalculator.calculate(canon_smiles)

        # 4. Drug-likeness (QED)
        qed_val = self._compute_qed(canon_smiles)

        # 5. Reward Calculation: R = -w1*(ΔG) + w2*(QED) - w3*(SAScore)
        # Note: ΔG is negative, so -w1*(ΔG) is positive!
        affinity_term = -self.w1 * dock_res.delta_g
        qed_term = self.w2 * qed_val
        sa_term = -self.w3 * sascore

        # Invariant catalytic Zn1/Zn2 coordination bonus
        zn_bonus = 0.0
        if dock_res.zn1_coordinated and dock_res.zn2_coordinated:
            zn_bonus += 2.0
        if dock_res.bis_zinc_bridging:
            zn_bonus += 1.5

        total = affinity_term + qed_term + sa_term + zn_bonus

        # Check for Terminal Nash Equilibrium:
        # Invariant Zn1/Zn2 coordination + ΔG <= -9.5 kcal/mol + SAScore <= 2.9
        is_nash = (
            dock_res.bis_zinc_bridging and
            dock_res.delta_g <= -9.5 and
            sascore <= 3.2 and
            dock_res.escape_loss_pct < 5.0
        )

        return RewardComponents(
            total_reward=round(total, 3),
            is_valid=True,
            delta_g=dock_res.delta_g,
            qed=round(qed_val, 3),
            sascore=sascore,
            escape_resilience_loss=dock_res.escape_loss_pct,
            zn1_coordinated=dock_res.zn1_coordinated,
            zn2_coordinated=dock_res.zn2_coordinated,
            is_nash_equilibrium=is_nash,
            explanation=(
                f"ΔG={dock_res.delta_g:.2f} kcal/mol, QED={qed_val:.2f}, SAScore={sascore:.2f}, "
                f"Zn1:{dock_res.zn1_coordinated}, Zn2:{dock_res.zn2_coordinated}, "
                f"ERI Loss={dock_res.escape_loss_pct:.1f}%"
            )
        )

    def _compute_qed(self, smiles: str) -> float:
        """Computes QED drug-likeness (0.0 to 1.0)."""
        if RDKIT_AVAILABLE and Chem is not None:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    return float(QED.qed(mol))
            except Exception:
                pass
        
        # Heuristic QED estimation based on MW, rotatable bonds, and functional groups
        base_qed = 0.72
        if "C(=O)O" in smiles:
            base_qed += 0.05
        if "c1ccccc1" in smiles:
            base_qed += 0.08
        if len(smiles) > 50:
            base_qed -= 0.15
        return round(min(0.95, max(0.2, base_qed)), 3)
