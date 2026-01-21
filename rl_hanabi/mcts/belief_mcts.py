"""
Belief-Integrated Monte Carlo Tree Search (MCTS) for Hanabi.

This MCTS implementation integrates the belief state and neural network model
to guide search. Instead of expanding all legal moves, it uses the model's
policy head to focus on promising moves and the value head for leaf evaluation.
"""

import math
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import torch

from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.belief.belief_state import BeliefState
from rl_hanabi.model.belief_model import ActionDecoder
from hanabi_learning_environment.pyhanabi import HanabiMove


@dataclass
class SearchTransition:
    """
    A transition collected during MCTS search ("thinking" transition).
    
    These transitions represent states explored during search that weren't
    actually played in the game, but provide additional training signal
    about what the model's policy and value should be at those states.
    """
    # State information (same format as game transitions)
    slot_beliefs: np.ndarray          # [P, H, C+R] - belief state
    affected_mask: np.ndarray         # [P, H] - mask for last action
    move_target_player: int           # target player offset
    acting_player: int                # acting player offset  
    action: np.ndarray                # [action_dim] - last action encoding
    fireworks: np.ndarray             # [C] - current fireworks
    discard_pile: np.ndarray          # [C*R] - discard pile
    
    # Targets for training
    true_colors: np.ndarray           # [H] - true colors of observer's hand
    true_ranks: np.ndarray            # [H] - true ranks of observer's hand
    legal_moves_mask: np.ndarray      # [action_space_size] - mask of legal moves
    
    # MCTS-derived targets
    policy_prior: np.ndarray          # [action_space_size] - neural network policy
    value_estimate: float             # Value estimate from neural network
    
    # Visit count info (if node was visited multiple times)
    visit_count: int = 1              # How many times this node was visited
    
    # Metadata
    game_config: Optional[dict] = None  # Game configuration
    search_depth: int = 0             # Depth in search tree


logger = logging.getLogger(__name__)


class BeliefNode:
    """
    Node for belief-integrated MCTS.
    
    Each node maintains:
    - The game state
    - The belief state for the current player
    - Visit counts and values for PUCT selection
    - Prior probabilities from the policy network
    """
    
    def __init__(
        self,
        state: HLEGameState,
        belief_state: BeliefState,
        last_action: Optional[HanabiMove],
        parent: Optional['BeliefNode'],
        prior: float = 0.0,
        c_puct: float = 1.4,
    ):
        self.state = state
        self.belief_state = belief_state
        self.last_action = last_action
        self.parent = parent
        self.prior = prior
        self.c_puct = c_puct
        
        # Tree statistics
        self.n_visits: int = 0
        self.value_sum: float = 0.0
        
        # Children management
        self.children: dict[int, BeliefNode] = {}  # action_index -> child node
        self.valid_moves = list(state.legal_moves())
        self.valid_moves_mask = state.legal_moves_mask()
        
        # Policy priors (set by expand())
        self.policy_priors: Optional[np.ndarray] = None
        self.is_expanded: bool = False
        
    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()
    
    @property
    def q_value(self) -> float:
        """Returns the mean value Q(s,a) = W(s,a) / N(s,a)"""
        if self.n_visits == 0:
            return 0.0
        return self.value_sum / self.n_visits
    
    def puct_score(self, parent_visits: int) -> float:
        """
        Compute PUCT score for selection.
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        exploration = self.c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.n_visits)
        return self.q_value + exploration
    
    def select_child(self) -> Tuple[int, 'BeliefNode']:
        """Select the child with highest PUCT score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action_idx, child in self.children.items():
            score = child.puct_score(self.n_visits)
            if score > best_score:
                best_score = score
                best_action = action_idx
                best_child = child
        
        if best_child is None:
            raise ValueError("No children to select from")
        
        return best_action, best_child
    
    def get_unexpanded_actions(self) -> List[int]:
        """Get list of valid action indices that haven't been expanded yet."""
        unexpanded = []
        for i, is_valid in enumerate(self.valid_moves_mask):
            if is_valid and i not in self.children:
                unexpanded.append(i)
        return unexpanded


class BeliefMCTS:
    """
    Belief-integrated MCTS for Hanabi.
    
    Uses the ActionDecoder model to:
    1. Get policy priors for move selection (policy head)
    2. Evaluate leaf nodes (value head)
    3. Update belief states as moves are applied
    
    Key features:
    - Progressive widening: Only expand promising moves based on policy
    - Belief tracking: Maintain belief states throughout the tree
    - Value-guided backup: Use neural network value estimates
    """
    
    def __init__(
        self,
        model: ActionDecoder,
        device: torch.device,
        time_ms: int = 1000,
        c_puct: float = 1.4,
        num_simulations: Optional[int] = None,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        top_k_actions: int = 10,  # Only expand top-k actions by prior
        collect_search_transitions: bool = False,  # Collect transitions from search
        min_visits_for_search_transition: int = 1,  # Minimum visits to collect a transition
    ):
        """
        Args:
            model: ActionDecoder model with policy and value heads
            device: Device for model inference
            time_ms: Time budget in milliseconds (used if num_simulations is None)
            c_puct: PUCT exploration constant
            num_simulations: Fixed number of simulations (overrides time_ms if set)
            temperature: Temperature for final action selection
            dirichlet_alpha: Dirichlet noise alpha for root exploration
            dirichlet_weight: Weight for Dirichlet noise at root
            top_k_actions: Only expand top-k actions by policy prior
            collect_search_transitions: If True, collect transitions from states visited during search
            min_visits_for_search_transition: Minimum visit count for a node to be collected
        """
        self.model = model
        self.device = device
        self.time_ms = time_ms
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self.top_k_actions = top_k_actions
        
        # Model dimensions for padding
        self.max_num_colors = model.max_num_colors
        self.max_num_ranks = model.max_num_ranks
        self.max_hand_size = model.max_hand_size
        self.max_num_players = model.max_num_players
        
        # Search transition collection settings
        self.collect_search_transitions = collect_search_transitions
        self.min_visits_for_search_transition = min_visits_for_search_transition
        self.search_transitions: List[SearchTransition] = []
        self._game_config: Optional[dict] = None  # Set when collecting transitions
        
        self.root: Optional[BeliefNode] = None
        
    def _pad_observation(
        self,
        all_hands: np.ndarray,
        fireworks: np.ndarray,
        discard_pile: np.ndarray,
        affected_mask: np.ndarray,
        belief_state: BeliefState,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pad observations to match model's expected max dimensions."""
        num_players = belief_state.num_players
        num_colors = belief_state.num_colors
        num_ranks = belief_state.num_ranks
        hand_size = belief_state.hand_size
        
        # Pad slot_beliefs: [P, H, C+R] -> [max_P, max_H, max_C + max_R]
        padded_beliefs = np.zeros(
            (self.max_num_players, self.max_hand_size, self.max_num_colors + self.max_num_ranks),
            dtype=np.float32
        )
        padded_beliefs[:num_players, :hand_size, :num_colors] = all_hands[:, :, :num_colors]
        padded_beliefs[:num_players, :hand_size, self.max_num_colors:self.max_num_colors + num_ranks] = \
            all_hands[:, :, num_colors:num_colors + num_ranks]
        
        # Pad fireworks: [C] -> [max_C]
        padded_fireworks = np.zeros(self.max_num_colors, dtype=np.float32)
        padded_fireworks[:num_colors] = fireworks
        
        # Pad discard_pile: [C*R] -> [max_C * max_R]
        padded_discard = np.zeros(self.max_num_colors * self.max_num_ranks, dtype=np.float32)
        for orig_idx in range(len(discard_pile)):
            if discard_pile[orig_idx] > 0:
                orig_color = orig_idx // num_ranks
                orig_rank = orig_idx % num_ranks
                new_idx = orig_color * self.max_num_ranks + orig_rank
                padded_discard[new_idx] = discard_pile[orig_idx]
        
        # Pad affected_mask: [P, H] -> [max_P, max_H]
        padded_mask = np.zeros((self.max_num_players, self.max_hand_size), dtype=np.float32)
        padded_mask[:num_players, :hand_size] = affected_mask[:num_players, :hand_size]
        
        return padded_beliefs, padded_fireworks, padded_discard, padded_mask
    
    @torch.no_grad()
    def _evaluate(self, belief_state: BeliefState, valid_moves_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluate a state using the neural network.
        
        Returns:
            policy: Normalized policy distribution over actions
            value: Estimated value in [0, 1]
        """
        player = belief_state.player
        num_players = belief_state.num_players
        
        # Prepare observations
        all_hands, fireworks, discard_pile_one_hot, tokens = belief_state.prepare_belief_obs(player)
        action_encoding, affected_mask = belief_state.encode_last_action()
        
        # Pad observations
        padded_beliefs, padded_fireworks, padded_discard, padded_mask = self._pad_observation(
            all_hands, fireworks, discard_pile_one_hot, affected_mask, belief_state
        )
        
        # Get player offsets
        player_idx, target_idx = belief_state.get_last_player_and_target_index()
        player_offset = (player_idx - player) % num_players
        target_offset = (target_idx - player) % num_players
        
        # Convert to tensors
        slot_beliefs_tensor = torch.from_numpy(padded_beliefs).float().unsqueeze(0).to(self.device)
        affected_mask_tensor = torch.from_numpy(padded_mask).float().unsqueeze(0).to(self.device)
        action_tensor = torch.from_numpy(action_encoding).float().unsqueeze(0).to(self.device)
        fireworks_tensor = torch.from_numpy(padded_fireworks).float().unsqueeze(0).to(self.device)
        discard_pile_tensor = torch.from_numpy(padded_discard).float().unsqueeze(0).to(self.device)
        target_player_tensor = torch.tensor([target_offset], dtype=torch.long, device=self.device)
        acting_player_tensor = torch.tensor([player_offset], dtype=torch.long, device=self.device)
        
        # Forward pass
        _, _, action_logits, value = self.model(
            slot_beliefs=slot_beliefs_tensor,
            affected_mask=affected_mask_tensor,
            move_target_player=target_player_tensor,
            acting_player=acting_player_tensor,
            action=action_tensor,
            fireworks=fireworks_tensor,
            discard_pile=discard_pile_tensor,
        )
        
        # Process policy
        action_logits = action_logits.squeeze(0).cpu().numpy()
        action_logits = action_logits[:len(valid_moves_mask)]
        action_logits[~valid_moves_mask] = -float('inf')
        
        # Softmax to get probabilities
        exp_logits = np.exp(action_logits - np.max(action_logits))
        policy = exp_logits / (exp_logits.sum() + 1e-10)
        
        return policy, value.item()
    
    def _create_child_belief_state(
        self,
        parent_belief: BeliefState,
        parent_state: HLEGameState,
        new_state: HLEGameState,
    ) -> BeliefState:
        """
        Create a new belief state for a child node after a move.
        """
        # Create new belief state for the new current player
        new_player = new_state.current_player_index
        child_belief = BeliefState(new_state, player=new_player)
        
        # Update belief from the move that was just made
        child_belief.update_from_move(model=self.model)
        
        return child_belief
    
    def _expand_node(self, node: BeliefNode) -> float:
        """
        Expand a node by evaluating it with the neural network.
        
        Returns the value estimate for backpropagation.
        """
        if node.is_terminal:
            # Terminal node - return actual score
            return node.state.score() / node.state.max_score()
        
        # Evaluate the node
        policy, value = self._evaluate(node.belief_state, node.valid_moves_mask)
        node.policy_priors = policy
        node.is_expanded = True
        
        # Only create children for top-k actions (progressive widening)
        valid_indices = np.where(node.valid_moves_mask)[0]
        if len(valid_indices) > 0:
            # Sort by policy prior and take top-k
            sorted_indices = sorted(valid_indices, key=lambda i: policy[i], reverse=True)
            top_indices = sorted_indices[:min(self.top_k_actions, len(sorted_indices))]
            
            for action_idx in top_indices:
                prior = float(policy[action_idx])
                if prior > 1e-6:  # Only expand if prior is non-negligible
                    # Create child state
                    child_state = node.state.copy()
                    move = child_state.index_to_move(action_idx)
                    child_state.apply_move(move)
                    
                    # Create child belief state
                    child_belief = self._create_child_belief_state(
                        node.belief_state, node.state, child_state
                    )
                    
                    # Create child node
                    child_node = BeliefNode(
                        state=child_state,
                        belief_state=child_belief,
                        last_action=move,
                        parent=node,
                        prior=prior,
                        c_puct=self.c_puct,
                    )
                    node.children[action_idx] = child_node
        
        return value
    
    def _add_dirichlet_noise(self, node: BeliefNode):
        """Add Dirichlet noise to root node priors for exploration."""
        if node.policy_priors is None:
            return
        
        valid_indices = list(node.children.keys())
        if len(valid_indices) == 0:
            return
        
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_indices))
        
        for i, action_idx in enumerate(valid_indices):
            child = node.children[action_idx]
            child.prior = (1 - self.dirichlet_weight) * child.prior + self.dirichlet_weight * noise[i]
    
    def _select(self, node: BeliefNode) -> BeliefNode:
        """Select a leaf node for expansion using PUCT."""
        while node.is_expanded and not node.is_terminal and len(node.children) > 0:
            _, node = node.select_child()
        return node
    
    def _backpropagate(self, node: BeliefNode, value: float):
        """Backpropagate value through the tree."""
        while node is not None:
            node.n_visits += 1
            node.value_sum += value
            node = node.parent # type: ignore
    
    def _create_search_transition(
        self, 
        node: BeliefNode, 
        policy: np.ndarray, 
        value: float,
        depth: int = 0,
    ) -> Optional[SearchTransition]:
        """
        Create a SearchTransition from a node that was visited during MCTS.
        
        Args:
            node: The BeliefNode to create a transition from
            policy: The policy priors from neural network evaluation
            value: The value estimate from neural network evaluation  
            depth: Depth in the search tree
            
        Returns:
            SearchTransition or None if creation fails
        """
        if self._game_config is None:
            return None
            
        try:
            belief_state = node.belief_state
            state = node.state
            player = belief_state.player
            num_players = belief_state.num_players
            
            # Get observations from belief state
            all_hands, fireworks, discard_pile_one_hot, tokens = belief_state.prepare_belief_obs(player)
            action_encoding, affected_mask = belief_state.encode_last_action()
            
            # Get ground truth for observer's hand
            observer_hand = state.get_hands()[player]
            true_colors = np.array([card.color() for card in observer_hand], dtype=np.int64)
            true_ranks = np.array([card.rank() for card in observer_hand], dtype=np.int64)
            
            # Get player offsets
            player_idx, target_idx = belief_state.get_last_player_and_target_index()
            player_offset = (player_idx - player) % num_players
            target_offset = (target_idx - player) % num_players
            
            # Pad ground truth if needed
            hand_size = self._game_config.get("hand_size", len(true_colors))
            if len(true_colors) < hand_size:
                pad_size = hand_size - len(true_colors)
                true_colors = np.pad(true_colors, (0, pad_size), constant_values=-1)
                true_ranks = np.pad(true_ranks, (0, pad_size), constant_values=-1)
            
            return SearchTransition(
                slot_beliefs=all_hands,
                affected_mask=affected_mask,
                move_target_player=target_offset,
                acting_player=player_offset,
                action=action_encoding,
                fireworks=fireworks,
                discard_pile=discard_pile_one_hot,
                true_colors=true_colors,
                true_ranks=true_ranks,
                legal_moves_mask=node.valid_moves_mask.copy(),
                policy_prior=policy.copy(),
                value_estimate=value,
                visit_count=node.n_visits,
                game_config=self._game_config.copy(),
                search_depth=depth,
            )
        except Exception as e:
            logger.debug(f"Error creating search transition: {e}")
            return None
    
    def _collect_tree_transitions(self, node: BeliefNode, depth: int = 0):
        """
        Recursively collect transitions from all nodes in the search tree.
        
        Only collects from nodes that:
        - Have been expanded (have policy priors)
        - Meet minimum visit count threshold
        - Are not the root (root is handled separately as actual game transition)
        """
        if not self.collect_search_transitions:
            return
            
        # Don't collect from root (that's the actual game transition)
        # and don't collect from terminal nodes
        if node.is_terminal:
            return
            
        # Only collect if node has been expanded and visited enough
        if (node.is_expanded 
            and node.policy_priors is not None 
            and node.n_visits >= self.min_visits_for_search_transition
            and node.parent is not None):  # Skip root
            
            transition = self._create_search_transition(
                node=node,
                policy=node.policy_priors,
                value=node.q_value,  # Use Q-value as value target (backed up from search)
                depth=depth,
            )
            if transition is not None:
                self.search_transitions.append(transition)
        
        # Recursively collect from children
        for child in node.children.values():
            self._collect_tree_transitions(child, depth + 1)
    
    def get_search_transitions(self) -> List[SearchTransition]:
        """Get all search transitions collected during the last run."""
        return self.search_transitions
    
    def clear_search_transitions(self):
        """Clear collected search transitions."""
        self.search_transitions = []
    
    def init_root(
        self, 
        state: HLEGameState, 
        belief_states: List[BeliefState],
        game_config: Optional[dict] = None,
    ):
        """
        Initialize the root node with the current game state.
        
        Args:
            state: Current game state
            belief_states: List of belief states for each player (use current player's)
            game_config: Game configuration dict (needed for search transition collection)
        """
        current_player = state.current_player_index
        belief_state = belief_states[current_player]
        
        # Store game config for search transition creation
        self._game_config = game_config
        
        # Clear search transitions from previous run
        self.clear_search_transitions()
        
        self.root = BeliefNode(
            state=state,
            belief_state=belief_state,
            last_action=None,
            parent=None,
            prior=1.0,
            c_puct=self.c_puct,
        )
    
    def run(self) -> Tuple[np.ndarray, float]:
        """
        Run MCTS search from the root node.
        
        Returns:
            policy: Action probability distribution based on visit counts
            value: Estimated value of the root state
        """
        if self.root is None:
            raise ValueError("Root node not initialized. Call init_root first.")
        
        if self.root.is_terminal:
            # Return uniform policy and actual score for terminal state
            policy = np.zeros(self.root.state.action_space_size, dtype=np.float32)
            return policy, self.root.state.score() / self.root.state.max_score()
        
        # Expand root first
        root_value = self._expand_node(self.root)
        self._backpropagate(self.root, root_value)
        
        # Add exploration noise at root
        self._add_dirichlet_noise(self.root)
        
        # Run simulations
        start_time = time.time()
        num_sims = 0
        
        while True:
            # Check stopping condition
            if self.num_simulations is not None:
                if num_sims >= self.num_simulations:
                    break
            else:
                if (time.time() - start_time) * 1000 >= self.time_ms:
                    break
            
            # Select
            leaf = self._select(self.root)
            
            # Expand and evaluate
            if leaf.is_terminal:
                value = leaf.state.score() / leaf.state.max_score()
            elif not leaf.is_expanded:
                value = self._expand_node(leaf)
            else:
                # Already expanded but no children (shouldn't happen often)
                value = leaf.q_value
            
            # Backpropagate
            self._backpropagate(leaf, value)
            num_sims += 1
        
        # Compute output policy from visit counts
        action_space_size = self.root.state.action_space_size
        visit_counts = np.zeros(action_space_size, dtype=np.float32)
        
        for action_idx, child in self.root.children.items():
            visit_counts[action_idx] = child.n_visits
        
        # Apply temperature
        if self.temperature == 0:
            # Greedy selection
            policy = np.zeros(action_space_size, dtype=np.float32)
            best_action = np.argmax(visit_counts)
            policy[best_action] = 1.0
        else:
            # Softmax with temperature
            visit_counts_temp = visit_counts ** (1.0 / self.temperature)
            total = visit_counts_temp.sum()
            if total > 0:
                policy = visit_counts_temp / total
            else:
                # Fallback to uniform over valid moves
                policy = self.root.valid_moves_mask.astype(np.float32)
                policy /= policy.sum() + 1e-10
        
        logger.debug(f"MCTS completed {num_sims} simulations in {(time.time() - start_time)*1000:.1f}ms")
        
        # Collect search transitions from the tree (if enabled)
        if self.collect_search_transitions:
            self._collect_tree_transitions(self.root, depth=0)
            logger.debug(f"Collected {len(self.search_transitions)} search transitions")
        
        return policy, self.root.q_value
    
    def select_action(self) -> Tuple[int, HanabiMove]:
        """
        Select the best action based on MCTS results.
        
        Returns:
            action_idx: Index of selected action
            move: The selected HanabiMove
        """
        if self.root is None:
            raise ValueError("Root node not initialized")
        
        # Get visit counts
        action_space_size = self.root.state.action_space_size
        visit_counts = np.zeros(action_space_size, dtype=np.float32)
        
        for action_idx, child in self.root.children.items():
            visit_counts[action_idx] = child.n_visits
        
        # Select based on temperature
        if self.temperature == 0:
            action_idx = int(np.argmax(visit_counts))
        else:
            visit_counts_temp = visit_counts ** (1.0 / self.temperature)
            total = visit_counts_temp.sum()
            if total > 0:
                probs = visit_counts_temp / total
                action_idx = np.random.choice(len(probs), p=probs)
            else:
                # Fallback to random valid move
                valid_indices = np.where(self.root.valid_moves_mask)[0]
                action_idx = int(np.random.choice(valid_indices))
        
        move = self.root.state.index_to_move(action_idx)
        return action_idx, move
    
    def advance_root(self, action_idx: int) -> bool:
        """
        Advance the root to a child node (for tree reuse).
        
        Args:
            action_idx: The action that was taken
            
        Returns:
            True if successfully reused subtree, False otherwise
        """
        if self.root is None or action_idx not in self.root.children:
            return False
        
        self.root = self.root.children[action_idx]
        self.root.parent = None  # Detach from old tree
        return True
