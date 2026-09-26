"""
Game State Representation for Two-Player Zero-Sum Adversarial Drug Design.

State S_t = (NDM-1 Active Site Pocket State, SMILES Lead, Turn, History)
- Player 0: Chemist Agent (White) - maximizes multi-objective reward
- Player 1: Pathogen Agent (Black) - minimizes Chemist reward via viable escape mutations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.pathogen.ndm1_target import PocketState, Mutation


@dataclass
class GameMove:
    player: str  # 'chemist' or 'pathogen'
    action_type: str
    details: Dict[str, Any]
    description: str


@dataclass
class GameState:
    pocket: PocketState
    smiles: str
    turn: int = 0  # 0: Chemist, 1: Pathogen
    step_count: int = 0
    max_steps: int = 6
    history: List[GameMove] = field(default_factory=list)
    is_terminal: bool = False
    terminal_reason: Optional[str] = None

    def clone(self) -> 'GameState':
        """Creates a deep copy of the game state."""
        return GameState(
            pocket=PocketState(mutations=dict(self.pocket.mutations)),
            smiles=self.smiles,
            turn=self.turn,
            step_count=self.step_count,
            max_steps=self.max_steps,
            history=list(self.history),
            is_terminal=self.is_terminal,
            terminal_reason=self.terminal_reason
        )

    def apply_chemist_move(self, new_smiles: str, action_desc: str, action_id: str) -> 'GameState':
        """Applies a chemical modification proposed by the Chemist Agent."""
        new_state = self.clone()
        new_state.smiles = new_smiles
        new_state.history.append(GameMove(
            player="chemist",
            action_type=action_id,
            details={"smiles": new_smiles},
            description=action_desc
        ))
        new_state.turn = 1  # Next turn: Pathogen
        new_state.step_count += 1
        if new_state.step_count >= new_state.max_steps:
            new_state.is_terminal = True
            new_state.terminal_reason = "Max step limit reached"
        return new_state

    def apply_pathogen_move(self, mut: Mutation, delta_fitness: float) -> 'GameState':
        """Applies an active-site escape mutation proposed by the Pathogen Agent."""
        new_state = self.clone()
        new_state.pocket = new_state.pocket.apply_mutation(mut)
        new_state.history.append(GameMove(
            player="pathogen",
            action_type="escape_mutation",
            details={"mutation": mut.code, "delta_fitness": delta_fitness},
            description=f"Pathogen introduced escape mutation {mut.code} (ΔFitness={delta_fitness:.2f})"
        ))
        new_state.turn = 0  # Next turn: Chemist
        new_state.step_count += 1
        if new_state.step_count >= new_state.max_steps:
            new_state.is_terminal = True
            new_state.terminal_reason = "Max step limit reached"
        return new_state
