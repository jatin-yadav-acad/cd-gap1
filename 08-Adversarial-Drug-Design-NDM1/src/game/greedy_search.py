"""
Greedy Best-First Search Baseline Policy.

Contrasts with MCTS-UCT:
- Evaluates candidate modifications greedily at the current step.
- Lacks adversarial tree lookahead and future-state anticipation.
- Prone to local optima and vulnerable to multi-residue pathogen escape trajectories.
"""

from typing import List, Tuple
from src.game.game_state import GameState
from src.pathogen.esm2_scanner import ESM2MutationalScanner
from src.pathogen.ndm1_target import Mutation
from src.chemist.llm_policy import ChemistReasoningPolicy, LLMProposal
from src.oracle.reward import MultiObjectiveRewardEngine, RewardComponents


class GreedySearchBaseline:
    """Greedy Best-First Search policy for drug design."""

    def __init__(
        self,
        chemist_policy: ChemistReasoningPolicy,
        pathogen_scanner: ESM2MutationalScanner,
        reward_engine: MultiObjectiveRewardEngine,
        num_candidates_per_step: int = 4
    ):
        self.chemist_policy = chemist_policy
        self.pathogen_scanner = pathogen_scanner
        self.reward_engine = reward_engine
        self.num_candidates_per_step = num_candidates_per_step

    def select_chemist_move(self, state: GameState) -> Tuple[GameState, RewardComponents]:
        """Greedily selects the highest immediate reward candidate."""
        best_state = state
        best_reward_comp: RewardComponents = self.reward_engine.evaluate(state.smiles, state.pocket)
        best_score = best_reward_comp.total_reward

        for _ in range(self.num_candidates_per_step):
            proposal: LLMProposal = self.chemist_policy.generate_candidate(
                state.smiles, state.pocket, state.step_count
            )
            candidate_state = state.apply_chemist_move(
                new_smiles=proposal.proposed_smiles,
                action_desc=proposal.rationale,
                action_id=proposal.intended_action
            )
            comp = self.reward_engine.evaluate(proposal.proposed_smiles, state.pocket)
            if comp.total_reward > best_score:
                best_score = comp.total_reward
                best_state = candidate_state
                best_reward_comp = comp

        return best_state, best_reward_comp

    def select_pathogen_move(self, state: GameState) -> GameState:
        """Pathogen greedily selects the escape mutation causing greatest affinity loss."""
        viable = self.pathogen_scanner.get_viable_mutations(state.pocket)
        if not viable:
            return state

        curr_eval = self.reward_engine.evaluate(state.smiles, state.pocket)
        worst_loss = -float('inf')
        worst_mut = viable[0][0]
        worst_fit = viable[0][1]

        for mut, fit in viable:
            test_pocket = state.pocket.apply_mutation(mut)
            test_eval = self.reward_engine.evaluate(state.smiles, test_pocket)
            loss = curr_eval.delta_g - test_eval.delta_g  # positive when affinity weakens
            if loss > worst_loss:
                worst_loss = loss
                worst_mut = mut
                worst_fit = fit

        return state.apply_pathogen_move(worst_mut, worst_fit)
