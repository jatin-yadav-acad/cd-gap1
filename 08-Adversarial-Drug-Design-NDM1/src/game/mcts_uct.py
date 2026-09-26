"""
Monte Carlo Tree Search with Upper Confidence Bounds applied to Trees (MCTS-UCT).

Implements the two-player zero-sum adversarial game tree search:
- Chemist Node (White): Maximizes multi-objective reward R = -w_1(ΔG) + w_2(QED) - w_3(SAScore).
- Pathogen Node (Black): Minimizes Chemist reward via viable escape mutations (ΔFitness > -1.5).
- UCT Formula:
      UCT(s, a) = Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))
- Converges to terminal Nash equilibrium where candidate molecule coordinates catalytic Zn1/Zn2.
"""

import math
import random
from typing import Dict, List, Optional, Tuple
from src.game.game_state import GameState, GameMove
from src.pathogen.esm2_scanner import ESM2MutationalScanner
from src.pathogen.ndm1_target import Mutation
from src.chemist.llm_policy import ChemistReasoningPolicy, LLMProposal
from src.oracle.reward import MultiObjectiveRewardEngine, RewardComponents


class MCTSNode:
    """Node in the adversarial MCTS game tree."""

    def __init__(self, state: GameState, parent: Optional['MCTSNode'] = None, action_taken: Optional[str] = None):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children: List['MCTSNode'] = []
        self.visits: int = 0
        self.total_value: float = 0.0
        self.reward_components: Optional[RewardComponents] = None
        self.untried_moves: Optional[List] = None

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def is_fully_expanded(self) -> bool:
        return self.untried_moves is not None and len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        if self.state.is_terminal:
            return True
        if self.reward_components and self.reward_components.is_nash_equilibrium:
            return True
        return False


class MCTSAdversarialSearch:
    """Adversarial MCTS-UCT search engine for closed-loop molecular evolution."""

    def __init__(
        self,
        chemist_policy: ChemistReasoningPolicy,
        pathogen_scanner: ESM2MutationalScanner,
        reward_engine: MultiObjectiveRewardEngine,
        c_param: float = 1.414,
        max_simulations: int = 40,
        rollout_depth: int = 4
    ):
        self.chemist_policy = chemist_policy
        self.pathogen_scanner = pathogen_scanner
        self.reward_engine = reward_engine
        self.c_param = c_param
        self.max_simulations = max_simulations
        self.rollout_depth = rollout_depth

    def search(self, root_state: GameState) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Executes MCTS-UCT simulations from the root state and returns the optimal move."""
        root = MCTSNode(state=root_state)
        root.reward_components = self.reward_engine.evaluate(root_state.smiles, root_state.pocket)

        for _ in range(self.max_simulations):
            # 1. Selection
            leaf = self._select(root)
            
            # 2. Expansion
            child = self._expand(leaf)
            
            # 3. Simulation / Rollout
            simulation_reward = self._simulate(child)
            
            # 4. Backpropagation
            self._backpropagate(child, simulation_reward)

        # Select child with highest visit count or highest Q-value
        best_child = max(root.children, key=lambda c: c.visits) if root.children else root
        return best_child, root.children

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selects leaf node traversing tree using UCT."""
        curr = node
        while not curr.is_terminal() and curr.is_fully_expanded() and curr.children:
            curr = self._select_best_child_uct(curr)
        return curr

    def _select_best_child_uct(self, node: MCTSNode) -> MCTSNode:
        """Applies UCT formula considering current player turn."""
        is_chemist_turn = (node.state.turn == 0)
        best_val = -float('inf') if is_chemist_turn else float('inf')
        best_child = node.children[0]

        log_parent = math.log(node.visits) if node.visits > 0 else 0.0

        for child in node.children:
            if child.visits == 0:
                return child
            
            exploitation = child.q_value
            exploration = self.c_param * math.sqrt(log_parent / child.visits)
            
            if is_chemist_turn:
                # Chemist maximizes reward
                uct_val = exploitation + exploration
                if uct_val > best_val:
                    best_val = uct_val
                    best_child = child
            else:
                # Pathogen minimizes chemist reward (zero-sum)
                uct_val = exploitation - exploration
                if uct_val < best_val:
                    best_val = uct_val
                    best_child = child

        return best_child

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expands node with candidate actions."""
        if node.is_terminal():
            return node

        if node.untried_moves is None:
            node.untried_moves = self._get_legal_moves(node.state)

        if not node.untried_moves:
            return node

        move = node.untried_moves.pop(random.randrange(len(node.untried_moves)))
        next_state = self._apply_move(node.state, move)
        child = MCTSNode(state=next_state, parent=node, action_taken=str(move))
        child.reward_components = self.reward_engine.evaluate(next_state.smiles, next_state.pocket)
        node.children.append(child)
        return child

    def _get_legal_moves(self, state: GameState) -> List:
        """Generates legal moves for the current player."""
        if state.turn == 0:
            # Chemist turn: query Reasoning LLM policy for candidate proposals
            proposals = []
            for _ in range(3):
                p = self.chemist_policy.generate_candidate(state.smiles, state.pocket, state.step_count)
                proposals.append(p)
            return proposals
        else:
            # Pathogen turn: get viable ESM-2 escape mutations
            viable_muts = self.pathogen_scanner.get_viable_mutations(state.pocket)
            # Pick top viable mutations
            return viable_muts[:4]

    def _apply_move(self, state: GameState, move) -> GameState:
        """Transitions state applying move."""
        if state.turn == 0:
            proposal: LLMProposal = move
            return state.apply_chemist_move(
                new_smiles=proposal.proposed_smiles,
                action_desc=proposal.rationale,
                action_id=proposal.intended_action
            )
        else:
            mut_tuple: Tuple[Mutation, float] = move
            mut, fitness = mut_tuple
            return state.apply_pathogen_move(mut=mut, delta_fitness=fitness)

    def _simulate(self, node: MCTSNode) -> float:
        """Rollout policy until horizon."""
        curr_state = node.state.clone()
        depth = 0
        while depth < self.rollout_depth and not curr_state.is_terminal:
            if curr_state.turn == 0:
                p = self.chemist_policy.generate_candidate(curr_state.smiles, curr_state.pocket, curr_state.step_count)
                curr_state = curr_state.apply_chemist_move(p.proposed_smiles, p.rationale, p.intended_action)
            else:
                viable = self.pathogen_scanner.get_viable_mutations(curr_state.pocket)
                if viable:
                    mut, score = random.choice(viable)
                    curr_state = curr_state.apply_pathogen_move(mut, score)
                else:
                    break
            depth += 1

        eval_res = self.reward_engine.evaluate(curr_state.smiles, curr_state.pocket)
        return eval_res.total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Updates visit counts and values up to root."""
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.total_value += reward
            curr = curr.parent
