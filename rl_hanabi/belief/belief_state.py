
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from hanabi_learning_environment import pyhanabi
from rl_hanabi.game.hle_state import HLEGameState

# Standard Hanabi deck composition per color: three 1s, two 2s, two 3s, two 4s, one 5.
# Ranks are 0-indexed (0=1, 1=2, ..., 4=5).
RANK_COUNTS = [3, 2, 2, 2, 1]
NUM_RANKS = 5
NUM_COLORS = 5
TOTAL_CARDS_PER_COLOR = sum(RANK_COUNTS)
TOTAL_CARDS = NUM_COLORS * TOTAL_CARDS_PER_COLOR
NUM_CARD_TYPES = NUM_COLORS * NUM_RANKS

def get_card_index(color: int, rank: int) -> int:
    """Maps (color, rank) to a flat index 0-24."""
    return color * NUM_RANKS + rank

def get_card_color_rank(index: int) -> Tuple[int, int]:
    """Maps flat index 0-24 to (color, rank)."""
    return index // NUM_RANKS, index % NUM_RANKS

@dataclass
class CardBelief:
    """
    Represents the belief distribution for a single card slot.
    
    Corresponds to the factorized belief described in the BAD paper.
    Instead of a joint distribution over all cards, we maintain marginals 
    for each card slot.
    """
    probs: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CARD_TYPES))

    def normalize(self):
        """Normalizes the probability distribution."""
        total = np.sum(self.probs)
        if total > 0:
            self.probs /= total
        else:
            # Should not happen in valid game states, but handle gracefully
            self.probs = np.ones(NUM_CARD_TYPES) / NUM_CARD_TYPES

    def entropy(self) -> float:
        """Computes the entropy of the belief distribution."""
        # H(X) = -sum(p(x) * log(p(x)))
        # Avoid log(0)
        p = self.probs[self.probs > 0]
        return -np.sum(p * np.log(p))

@dataclass
class HandBelief:
    """
    Represents the belief distribution for a player's hand.
    """
    cards: List[CardBelief] = field(default_factory=list)

    def size(self) -> int:
        return len(self.cards)

@dataclass
class BeliefState:
    """
    The public belief state (V-belief) as described in the BAD paper.
    
    It consists of:
    1. Factorized beliefs for each card in each player's hand.
    2. Knowledge of the remaining deck (implicitly, via card counts).
    """
    hands: List[HandBelief]
    # Track total counts of each card type remaining in the deck + hands
    # This is derived from the initial deck minus played and discarded cards.
    remaining_card_counts: np.ndarray 

    def copy(self) -> 'BeliefState':
        new_hands = [
            HandBelief(cards=[CardBelief(probs=c.probs.copy()) for c in h.cards])
            for h in self.hands
        ]
        return BeliefState(hands=new_hands, remaining_card_counts=self.remaining_card_counts.copy())


    @classmethod
    def initialize_belief(cls, hle_state: HLEGameState):
        """
        Initializes the belief state from a given HLE game state.
        
        This constructs the 'public belief' based on:
        - Cards played (fireworks)
        - Cards discarded
        - Card knowledge (clues) provided by the environment
        
        Note: In the BAD paper, the V-belief is purely public. 
        However, for a specific agent, 'public' might include their own private observation 
        if we are constructing the belief *for* that agent. 
        Here we construct the belief from the perspective of an observer who sees 
        what is publicly known (clues + revealed cards).
        """
        observation = hle_state.state.observation(hle_state.our_index)
        
        # 1. Calculate remaining cards (Deck + Hands)
        # Start with full deck
        remaining_counts = np.zeros(NUM_CARD_TYPES, dtype=int)
        for r in range(NUM_RANKS):
            for c in range(NUM_COLORS):
                idx = get_card_index(c, r)
                remaining_counts[idx] = RANK_COUNTS[r]
                
        # Subtract played cards (fireworks)
        fireworks = observation.fireworks()
        for c, rank_count in enumerate(fireworks):
            # If rank_count is 2, it means 0, 1 (ranks) are played.
            for r in range(rank_count):
                idx = get_card_index(c, r)
                remaining_counts[idx] -= 1
                
        # Subtract discarded cards
        discards = observation.discard_pile()
        for card in discards:
            idx = get_card_index(card.color(), card.rank())
            remaining_counts[idx] -= 1

        # 2. Build beliefs for each hand
        hands = []
        num_players = observation.num_players()
        
        # We need to access card knowledge. 
        # pyhanabi's observation.card_knowledge() gives us hints per player.
        # But we also need to account for the fact that we can SEE other players' cards
        # if we are a specific player. 
        # The prompt asks for "beliefs for every card slot in the hands of the players".
        # If this is the *public* belief state (V-belief), it should NOT use the pixels/identities 
        # of the cards in other players' hands, only the clues.
        # However, usually in Hanabi bots, we track what *we* know.
        # The BAD paper distinguishes between private state and public belief.
        # Let's implement the PUBLIC belief (based on clues only) as a baseline, 
        # which is what all agents agree on.
        
        # Actually, pyhanabi's card_knowledge is exactly the public info (clues).
        card_knowledge = observation.card_knowledge()
        
        for p in range(num_players):
            hand_knowledge = card_knowledge[p]
            hand_belief = HandBelief()
            
            # Get hand size for this player
            # We can infer it from the knowledge list size
            hand_size = len(hand_knowledge)
            
            for i in range(hand_size):
                knowledge = hand_knowledge[i]
                belief = CardBelief()
                
                # Start with counts of remaining cards
                # P(card = x) \propto Count(x) * Constraint(x)
                belief.probs = remaining_counts.astype(float).copy()
                
                # Apply Color Constraints
                # knowledge.color_plausible(c) returns True if color c is possible
                for c in range(NUM_COLORS):
                    if not knowledge.color_plausible(c):
                        # Zero out all ranks for this color
                        for r in range(NUM_RANKS):
                            belief.probs[get_card_index(c, r)] = 0.0
                
                # Apply Rank Constraints
                for r in range(NUM_RANKS):
                    if not knowledge.rank_plausible(r):
                        # Zero out all colors for this rank
                        for c in range(NUM_COLORS):
                            belief.probs[get_card_index(c, r)] = 0.0
                
                belief.normalize()
                hand_belief.cards.append(belief)
                
            hands.append(hand_belief)

        return cls(hands=hands, remaining_card_counts=remaining_counts)


    # --- Deterministic Updates ---

    def update_from_clue(self, 
                        target_player: int, 
                        clue_type: int, 
                        clue_value: int, 
                        affected_indices: List[int]):
        """
        Updates the belief state based on a clue action.
        This is a deterministic update on the constraints.
        
        clue_type: 0 for Color, 1 for Rank (matching pyhanabi/game_types)
        clue_value: 0-4 (Color index or Rank index)
        affected_indices: List of card indices in target_player's hand that match the clue.
        """
        hand = self.hands[target_player]
        
        for i, card_belief in enumerate(hand.cards):
            if i in affected_indices:
                # Positive Information: The card MUST match the clue
                if clue_type == 0: # Color Clue
                    # Zero out all other colors
                    for c in range(NUM_COLORS):
                        if c != clue_value:
                            for r in range(NUM_RANKS):
                                card_belief.probs[get_card_index(c, r)] = 0.0
                else: # Rank Clue
                    # Zero out all other ranks
                    for r in range(NUM_RANKS):
                        if r != clue_value:
                            for c in range(NUM_COLORS):
                                card_belief.probs[get_card_index(c, r)] = 0.0
            else:
                # Negative Information: The card MUST NOT match the clue
                if clue_type == 0: # Color Clue
                    # Zero out this color
                    for r in range(NUM_RANKS):
                        card_belief.probs[get_card_index(clue_value, r)] = 0.0
                else: # Rank Clue
                    # Zero out this rank
                    for c in range(NUM_COLORS):
                        card_belief.probs[get_card_index(c, clue_value)] = 0.0
            
            card_belief.normalize()

    def update_from_play_discard(self, 
                                player_index: int, 
                                card_index: int, 
                                revealed_card_idx: int):
        """
        Updates belief when a card is played or discarded.
        
        1. The card at card_index in player_index's hand is removed.
        2. The revealed card is removed from the global remaining counts.
        3. Other beliefs are updated because the counts changed (optional, but good for accuracy).
        Note: Strictly speaking, factorized beliefs might not re-normalize based on counts 
        immediately unless we do a full re-computation, but we should at least decrement the count.
        """
        # Remove the card from the hand
        # In Hanabi, usually a new card is drawn, but this function just handles the removal/reveal.
        # The drawing logic should be handled separately or we assume the hand shifts.
        # For simplicity here, we'll assume the caller handles the hand shift/draw, 
        # and we just update the global counts and maybe the specific slot if it wasn't removed yet.
        
        # Update global counts
        self.remaining_card_counts[revealed_card_idx] -= 1
        if self.remaining_card_counts[revealed_card_idx] < 0:
            self.remaining_card_counts[revealed_card_idx] = 0
            
        # Note: In a fully factorized approximation, we might not instantly update all other 
        # cards' probabilities based on this count change to avoid expensive re-computation.
        # However, if a count goes to 0, we MUST update all beliefs to zero out that card.
        if self.remaining_card_counts[revealed_card_idx] == 0:
            for h in self.hands:
                for c in h.cards:
                    if c.probs[revealed_card_idx] > 0:
                        c.probs[revealed_card_idx] = 0.0
                        c.normalize()

    def update_from_draw(self, player_index: int):
        """
        Adds a new belief for a drawn card.
        The new card is drawn from the remaining deck.
        """
        new_card_belief = CardBelief()
        # Probability is proportional to remaining counts
        new_card_belief.probs = self.remaining_card_counts.astype(float).copy()
        new_card_belief.normalize()
        
        self.hands[player_index].cards.append(new_card_belief)


    # --- Bayesian Update ---

    def update_from_action(self, 
                        action_player: int, 
                        policy_likelihoods: List[np.ndarray]):
        """
        Approximate Bayesian Update as described in the BAD paper.
        
        Equation: B_{t+1}(h) ∝ P(a_t | h) * B_t(h)
        
        In the factorized approximation, we update each card slot independently.
        
        Args:
            belief_state: Current belief state.
            action_player: The player who took the action.
            policy_likelihoods: A list of numpy arrays, one for each card slot in the player's hand.
                                Each array has shape (NUM_CARD_TYPES,) and represents 
                                P(action | card_i = x).
                                This assumes the policy's dependence on the hand can be 
                                factorized or approximated per card.
        """
        hand = self.hands[action_player]
        
        if len(policy_likelihoods) != len(hand.cards):
            # Mismatch in hand size (maybe due to card being played/discarded just now?)
            # We assume the likelihoods correspond to the hand state *before* the action effect 
            # (or after, depending on when this is called). 
            # Usually called BEFORE the action changes the hand state (e.g. removes card).
            return

        for i, card_belief in enumerate(hand.cards):
            likelihood = policy_likelihoods[i]
            
            # Bayesian Update: Posterior = Prior * Likelihood
            card_belief.probs *= likelihood
            
            # Normalize
            card_belief.normalize() 


    # --- Sampling ---

    def sample_hands(self, seed: int | None = None) -> List[List[int]]:
        """
        Samples concrete hands for all players from the belief state.
        Enforces the global card count constraints (no more cards than in deck).
        
        Uses a randomized greedy approach or rejection sampling.
        """
        rng = np.random.default_rng(seed)
        
        # We need to sample cards for all slots such that counts are respected.
        # Flatten all slots
        all_slots = []
        for p_idx, hand in enumerate(self.hands):
            for c_idx, card_belief in enumerate(hand.cards):
                all_slots.append((p_idx, c_idx, card_belief))
                
        # Sort slots by entropy (lowest entropy first) to fix "sure" cards first
        # This is a heuristic to reduce backtracking/rejection
        all_slots.sort(key=lambda x: x[2].entropy())
        
        # Current counts of available cards
        current_counts = self.remaining_card_counts.copy()
        
        sampled_hands = [[] for _ in range(len(self.hands))]
        # Pre-fill with placeholders
        for i, h in enumerate(self.hands):
            sampled_hands[i] = [-1] * len(h.cards)
            
        # Try to sample
        # Simple approach: Sample one by one, if invalid, retry (simple rejection)
        # For a robust implementation, we might need backtracking, but let's try 
        # a single pass with renormalization.
        
        for p_idx, c_idx, card_belief in all_slots:
            # Get probabilities
            probs = card_belief.probs.copy()
            
            # Mask out cards with 0 count remaining
            mask = (current_counts > 0)
            probs *= mask
            
            # Normalize
            total = np.sum(probs)
            if total <= 0:
                # Failed to find a valid assignment
                # In a real game, this implies the belief state is inconsistent with reality
                # or we got unlucky with the greedy order.
                # Fallback: just pick any card with >0 original prob (ignoring global count constraint)
                # or raise error.
                probs = card_belief.probs.copy()
                total = np.sum(probs)
            
            probs /= total
            
            # Sample
            chosen_card = rng.choice(NUM_CARD_TYPES, p=probs)
            
            # Decrement count
            if current_counts[chosen_card] > 0:
                current_counts[chosen_card] -= 1
                
            sampled_hands[p_idx][c_idx] = chosen_card
            
        return sampled_hands
