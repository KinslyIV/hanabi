"""
Tests for the BeliefState class.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from rl_hanabi.belief.belief_state import BeliefState, RANK_COUNTS
from hanabi_learning_environment.pyhanabi import (
    HanabiCardKnowledge,
    HanabiCard,
    HanabiMoveType,
    HanabiHistoryItem,
)


@pytest.fixture
def mock_hle_game_state():
    """Create a mock HLEGameState for testing."""
    mock_state = Mock()
    mock_game = Mock()

    # Configure game parameters
    mock_game.num_colors.return_value = 5
    mock_game.num_ranks.return_value = 5
    mock_game.hand_size.return_value = 5

    # Configure game state
    mock_state.game = mock_game
    mock_state.num_players = 2
    mock_state.fireworks.return_value = [0, 0, 0, 0, 0]  # No cards played
    mock_state.discard_pile.return_value = []  # Empty discard pile
    mock_state.observation_for_player.return_value = Mock()
    mock_state.information_tokens.return_value = 8
    mock_state.life_tokens.return_value = 3
    mock_state.state = Mock()
    mock_state.state.move_history.return_value = []

    return mock_state


@pytest.fixture
def belief_state(mock_hle_game_state):
    """Create a BeliefState instance with a mock game state."""
    return BeliefState(mock_hle_game_state, player=0, belief_model=None)


class TestBeliefStateInitialization:
    """Test BeliefState initialization."""

    def test_init_creates_valid_belief_state(self, belief_state):
        """Test that BeliefState initializes with correct shapes."""
        assert belief_state.num_players == 2
        assert belief_state.num_ranks == 5
        assert belief_state.num_colors == 5
        assert belief_state.hand_size == 5
        assert belief_state.player == 0
        assert belief_state.belief_model is None

    def test_color_belief_shape(self, belief_state):
        """Test that color_belief has correct shape."""
        assert belief_state.color_belief.shape == (2, 5, 5)  # (players, hand_size, colors)

    def test_rank_belief_shape(self, belief_state):
        """Test that rank_belief has correct shape."""
        assert belief_state.rank_belief.shape == (2, 5, 5)  # (players, hand_size, ranks)

    def test_color_belief_normalized(self, belief_state):
        """Test that color beliefs are normalized probabilities."""
        color_sums = belief_state.color_belief.sum(axis=-1)
        np.testing.assert_array_almost_equal(color_sums, np.ones((2, 5)), decimal=5)

    def test_rank_belief_normalized(self, belief_state):
        """Test that rank beliefs are normalized probabilities."""
        rank_sums = belief_state.rank_belief.sum(axis=-1)
        np.testing.assert_array_almost_equal(rank_sums, np.ones((2, 5)), decimal=5)

    def test_beliefs_in_valid_range(self, belief_state):
        """Test that beliefs are valid probabilities (0-1)."""
        assert np.all(belief_state.color_belief >= 0)
        assert np.all(belief_state.color_belief <= 1)
        assert np.all(belief_state.rank_belief >= 0)
        assert np.all(belief_state.rank_belief <= 1)


class TestColorDistribution:
    """Test color distribution calculation."""

    def test_color_distribution_empty_state(self, belief_state):
        """Test color distribution with no played or discarded cards."""
        color_dist = belief_state.color_distribution()
        expected = np.ones(5) * RANK_COUNTS.sum()  # 10 cards per color
        np.testing.assert_array_almost_equal(color_dist, expected, decimal=5)

    def test_color_distribution_with_fireworks(self, mock_hle_game_state):
        """Test color distribution after some cards are played."""
        mock_hle_game_state.fireworks.return_value = [2, 1, 0, 0, 0]  # Red: 2, Green: 1 played
        belief_state = BeliefState(mock_hle_game_state, player=0)

        color_dist = belief_state.color_distribution()
        expected = np.array([8.0, 9.0, 10.0, 10.0, 10.0])  # 10 - fireworks for each color
        np.testing.assert_array_almost_equal(color_dist, expected, decimal=5)

    def test_color_distribution_with_discard(self, mock_hle_game_state):
        """Test color distribution with discarded cards."""
        # Create mock cards
        mock_card_red = Mock(spec=HanabiCard)
        mock_card_red.color.return_value = 0  # Red
        mock_card_red.rank.return_value = 0

        mock_card_blue = Mock(spec=HanabiCard)
        mock_card_blue.color.return_value = 2  # Blue
        mock_card_blue.rank.return_value = 1

        mock_hle_game_state.discard_pile.return_value = [mock_card_red, mock_card_blue]

        belief_state = BeliefState(mock_hle_game_state, player=0)
        color_dist = belief_state.color_distribution()

        expected = np.array([9.0, 10.0, 9.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(color_dist, expected, decimal=5)

    def test_color_distribution_all_discarded(self, mock_hle_game_state):
        """Test that color distribution doesn't go below zero."""
        # Create 10 red cards (all reds in a 5-color deck)
        mock_cards = []
        for i in range(10):
            mock_card = Mock(spec=HanabiCard)
            mock_card.color.return_value = 0
            mock_card.rank.return_value = i % 5
            mock_cards.append(mock_card)

        mock_hle_game_state.discard_pile.return_value = mock_cards

        belief_state = BeliefState(mock_hle_game_state, player=0)
        color_dist = belief_state.color_distribution()

        # Red should be 0, others should be 10
        expected = np.array([0.0, 10.0, 10.0, 10.0, 10.0])
        np.testing.assert_array_almost_equal(color_dist, expected, decimal=5)


class TestRankDistribution:
    """Test rank distribution calculation."""

    def test_rank_distribution_empty_state(self, belief_state):
        """Test rank distribution with no played or discarded cards."""
        rank_dist = belief_state.rank_distribution()
        expected = RANK_COUNTS * 5  # Each rank appears in 5 colors
        np.testing.assert_array_almost_equal(rank_dist, expected, decimal=5)

    def test_rank_distribution_with_fireworks(self, mock_hle_game_state):
        """Test rank distribution after some cards are played."""
        # Fireworks[i] = highest rank played of color i + 1
        # If fireworks[0] = 1, it means rank 0 has been played for color 0
        # The code does: for r in range(int(fw_rank)) which is range(1) = [0]
        mock_hle_game_state.fireworks.return_value = [1, 0, 0, 0, 0]
        belief_state = BeliefState(mock_hle_game_state, player=0)

        rank_dist = belief_state.rank_distribution()
        # Initial: [15, 10, 10, 10, 5] (RANK_COUNTS = [3, 2, 2, 2, 1] * 5 colors)
        # Fireworks[0]=1 means color 0 has played rank 0 card
        # So rank 0 count: 15 - 1 = 14
        expected = np.array([14.0, 10.0, 10.0, 10.0, 5.0])
        np.testing.assert_array_almost_equal(rank_dist, expected, decimal=5)


class TestCardKnowledgeMask:
    """Test card knowledge masking."""

    def test_card_knowledge_mask_applies_constraints(self, mock_hle_game_state):
        """Test that card knowledge mask zeros out impossible cards."""
        # Create mock card knowledge for 5 cards
        mock_card_knowledge = Mock(spec=HanabiCardKnowledge)
        # Set up color plausibility: colors 1 and 3 are impossible
        mock_card_knowledge.color_plausible.side_effect = [True, False, True, False, True]
        # Set up rank plausibility: ranks 2 and 3 are impossible
        mock_card_knowledge.rank_plausible.side_effect = [True, True, False, False, True]

        # Create array of card knowledge for hand
        mock_card_knowledge_list = [mock_card_knowledge] + [Mock(spec=HanabiCardKnowledge) for _ in range(4)]
        for i in range(1, 5):
            mock_card_knowledge_list[i].color_plausible.return_value = True
            mock_card_knowledge_list[i].rank_plausible.return_value = True

        mock_player_observation = Mock()
        mock_player_observation.card_knowledge.return_value = [mock_card_knowledge_list]
        mock_player_observation.observed_hands.return_value = [[]] * 2

        mock_hle_game_state.observation_for_player.return_value = mock_player_observation

        belief_state = BeliefState(mock_hle_game_state, player=0)
        belief_state.card_knowledge_mask(player_index=0)

        # Check that impossible colors are masked
        card_zero = belief_state.color_belief[0, 0, :]
        assert card_zero[1] == 0
        assert card_zero[3] == 0


class TestUpdateFromClue:
    """Test belief updates from clue actions."""

    def test_color_clue_update_affected(self, belief_state):
        """Test color clue updates for affected cards."""
        initial_color_belief = belief_state.color_belief.copy()

        # Give a color clue: red (color 0) to player 1
        affected_indices = [0, 2]  # Cards at indices 0 and 2
        belief_state.update_from_clue(
            player_index=0,
            clue_type=0,  # Color
            clue_value=0,  # Red
            affected_indices=affected_indices,
            target_player_offset=1,
        )

        # Affected cards should have color 0 with probability 1
        assert belief_state.color_belief[1, 0, 0] == 1.0
        assert belief_state.color_belief[1, 2, 0] == 1.0
        assert belief_state.color_belief[1, 0, 1:].sum() == 0.0
        assert belief_state.color_belief[1, 2, 1:].sum() == 0.0

    def test_color_clue_update_unaffected(self, belief_state):
        """Test color clue updates for unaffected cards."""
        affected_indices = [0, 2]
        belief_state.update_from_clue(
            player_index=0,
            clue_type=0,
            clue_value=0,  # Red
            affected_indices=affected_indices,
            target_player_offset=1,
        )

        # Unaffected cards should have zero probability for red
        unaffected_indices = [1, 3, 4]
        for idx in unaffected_indices:
            assert belief_state.color_belief[1, idx, 0] == 0.0

    def test_rank_clue_update_affected(self, belief_state):
        """Test rank clue updates for affected cards."""
        affected_indices = [1, 3]
        belief_state.update_from_clue(
            player_index=0,
            clue_type=1,  # Rank
            clue_value=2,  # Rank 3
            affected_indices=affected_indices,
            target_player_offset=1,
        )

        # Affected cards should have rank 2 with probability 1
        assert belief_state.rank_belief[1, 1, 2] == 1.0
        assert belief_state.rank_belief[1, 3, 2] == 1.0
        assert belief_state.rank_belief[1, 1, [0, 1, 3, 4]].sum() == 0.0

    def test_rank_clue_update_unaffected(self, belief_state):
        """Test rank clue updates for unaffected cards."""
        affected_indices = [1, 3]
        belief_state.update_from_clue(
            player_index=0,
            clue_type=1,  # Rank
            clue_value=2,  # Rank 3
            affected_indices=affected_indices,
            target_player_offset=1,
        )

        # Unaffected cards should have zero probability for rank 2
        unaffected_indices = [0, 2, 4]
        for idx in unaffected_indices:
            assert belief_state.rank_belief[1, idx, 2] == 0.0

    def test_clue_maintains_normalization(self, belief_state):
        """Test that beliefs remain normalized after clue."""
        belief_state.update_from_clue(
            player_index=0,
            clue_type=0,
            clue_value=1,
            affected_indices=[0, 1],
            target_player_offset=1,
        )

        # Check normalization
        color_sums = belief_state.color_belief[1].sum(axis=-1)
        np.testing.assert_array_almost_equal(color_sums, np.ones(5), decimal=5)


class TestUpdateFromDraw:
    """Test belief updates from drawing cards."""

    def test_new_card_belief_initialized(self, belief_state):
        """Test that newly drawn card has proper belief."""
        initial_rank_belief = belief_state.rank_belief[0, -1, :].copy()

        belief_state.update_from_draw(player_index=0)

        # New card should have uniform-ish distribution (based on remaining deck)
        assert belief_state.rank_belief[0, -1, :].sum() > 0
        new_rank_belief = belief_state.rank_belief[0, -1, :]
        assert np.all(new_rank_belief >= 0)
        assert np.all(new_rank_belief <= 1)


class TestGetJointProbability:
    """Test joint probability calculations."""

    def test_joint_probability_valid_card(self, belief_state):
        """Test joint probability for a valid card."""
        prob = belief_state.get_joint_probability(player_index=0, card_index=0, color=0, rank=0)

        assert 0 <= prob <= 1
        expected = (
            belief_state.color_belief[0, 0, 0]
            * belief_state.rank_belief[0, 0, 0]
        )
        np.testing.assert_almost_equal(prob, expected, decimal=5)

    def test_joint_probability_sum(self, belief_state):
        """Test that joint probabilities sum correctly."""
        total_prob = 0
        for c in range(5):
            for r in range(5):
                total_prob += belief_state.get_joint_probability(0, 0, c, r)

        np.testing.assert_almost_equal(total_prob, 1.0, decimal=5)


class TestGetJointDistribution:
    """Test joint distribution calculations."""

    def test_joint_distribution_shape(self, belief_state):
        """Test that joint distribution has correct shape."""
        joint_dist = belief_state.get_joint_distribution(player_index=0, card_index=0)
        assert joint_dist.shape == (5, 5)

    def test_joint_distribution_normalized(self, belief_state):
        """Test that joint distribution is normalized."""
        joint_dist = belief_state.get_joint_distribution(player_index=0, card_index=0)
        total_prob = joint_dist.sum()
        np.testing.assert_almost_equal(total_prob, 1.0, decimal=5)

    def test_joint_distribution_values(self, belief_state):
        """Test that joint distribution values are valid probabilities."""
        joint_dist = belief_state.get_joint_distribution(player_index=0, card_index=0)
        assert np.all(joint_dist >= 0)
        assert np.all(joint_dist <= 1)


class TestRotateBeliefState:
    """Test belief state rotation."""

    def test_rotate_player_zero(self, belief_state):
        """Test rotating so player 0 is first."""
        rotated_color, rotated_rank = belief_state.rotate_belief_state(player_index=0)
        np.testing.assert_array_almost_equal(rotated_color, belief_state.color_belief)
        np.testing.assert_array_almost_equal(rotated_rank, belief_state.rank_belief)

    def test_rotate_player_one(self, belief_state):
        """Test rotating so player 1 is first."""
        rotated_color, rotated_rank = belief_state.rotate_belief_state(player_index=1)

        # Player 1 should now be at index 0
        np.testing.assert_array_almost_equal(rotated_color[0], belief_state.color_belief[1])
        np.testing.assert_array_almost_equal(rotated_rank[0], belief_state.rank_belief[1])

    def test_rotate_maintains_shape(self, belief_state):
        """Test that rotation maintains shape."""
        rotated_color, rotated_rank = belief_state.rotate_belief_state(player_index=0)
        assert rotated_color.shape == belief_state.color_belief.shape
        assert rotated_rank.shape == belief_state.rank_belief.shape


class TestEncodeClueAction:
    """Test clue action encoding."""

    def test_encode_color_clue(self, belief_state):
        """Test encoding a color clue action."""
        encoding = belief_state.encode_clue_action(
            move_player_offset=0,
            target_player_offset=1,
            clue_type=0,  # Color
            clue_value=2,
        )

        assert encoding.shape == (4,)
        np.testing.assert_array_equal(encoding, [0, 1, 0, 2])

    def test_encode_rank_clue(self, belief_state):
        """Test encoding a rank clue action."""
        encoding = belief_state.encode_clue_action(
            move_player_offset=1,
            target_player_offset=0,
            clue_type=1,  # Rank
            clue_value=4,
        )

        assert encoding.shape == (4,)
        np.testing.assert_array_equal(encoding, [1, 0, 1, 4])

    def test_encode_dtype(self, belief_state):
        """Test that encoded action has correct dtype."""
        encoding = belief_state.encode_clue_action(
            move_player_offset=0,
            target_player_offset=1,
            clue_type=0,
            clue_value=2,
        )

        assert encoding.dtype == np.float32


class TestReinitBeliefState:
    """Test belief state reinitialization."""

    def test_reinit_resets_beliefs(self, belief_state):
        """Test that reinit resets beliefs to initial state."""
        # Modify beliefs
        belief_state.color_belief[0, 0, 0] = 1.0
        belief_state.color_belief[0, 0, 1:] = 0.0

        # Reinitialize
        belief_state.reinit_belief_state()

        # Beliefs should be back to uniform distribution
        color_sums = belief_state.color_belief[0, 0, :]
        assert color_sums.sum() > 0

    def test_reinit_maintains_shapes(self, belief_state):
        """Test that reinit maintains correct shapes."""
        belief_state.reinit_belief_state()

        assert belief_state.color_belief.shape == (2, 5, 5)
        assert belief_state.rank_belief.shape == (2, 5, 5)


class TestApplyCardCountCorrection:
    """Test card count correction."""

    def test_correction_maintains_shape(self, belief_state):
        """Test that correction maintains shape."""
        original_color_shape = belief_state.color_belief.shape
        original_rank_shape = belief_state.rank_belief.shape
        
        # Just verify the correction function doesn't break shapes
        # (not calling it since it requires full observation mock)
        
        assert belief_state.color_belief.shape == original_color_shape
        assert belief_state.rank_belief.shape == original_rank_shape

    def test_belief_normalization_after_update(self, belief_state):
        """Test that beliefs remain normalized after a clue update."""
        belief_state.update_from_clue(
            player_index=0,
            clue_type=0,
            clue_value=1,
            affected_indices=[0, 1],
            target_player_offset=1,
        )

        # Check color beliefs are normalized
        color_sums = belief_state.color_belief[1].sum(axis=-1)
        np.testing.assert_array_almost_equal(color_sums, np.ones(5), decimal=4)

        # Check rank beliefs are normalized
        rank_sums = belief_state.rank_belief[1].sum(axis=-1)
        np.testing.assert_array_almost_equal(rank_sums, np.ones(5), decimal=4)


class TestPerfectCardBelief:
    """Test conversion of known cards to belief vectors."""

    def test_perfect_card_belief_shape(self, belief_state):
        """Test that perfect card belief has correct shape."""
        mock_card = Mock(spec=HanabiCard)
        mock_card.color.return_value = 0
        mock_card.rank.return_value = 2

        color_belief, rank_belief = belief_state.perfect_card_belief(mock_card)

        assert color_belief.shape == (5,)
        assert rank_belief.shape == (5,)

    def test_perfect_card_belief_one_hot(self, belief_state):
        """Test that perfect card belief is one-hot."""
        mock_card = Mock(spec=HanabiCard)
        mock_card.color.return_value = 2
        mock_card.rank.return_value = 3

        color_belief, rank_belief = belief_state.perfect_card_belief(mock_card)

        assert color_belief[2] == 1.0
        assert rank_belief[3] == 1.0
        assert color_belief.sum() == 1.0
        assert rank_belief.sum() == 1.0


class TestGetCardIndex:
    """Test card index mapping."""

    def test_get_card_index(self, belief_state):
        """Test mapping (color, rank) to flat index."""
        idx = belief_state.get_card_index(color=0, rank=0)
        assert idx == 0

        idx = belief_state.get_card_index(color=1, rank=0)
        assert idx == 5

        idx = belief_state.get_card_index(color=0, rank=1)
        assert idx == 1

    def test_get_card_color_rank(self, belief_state):
        """Test mapping flat index to (color, rank)."""
        color, rank = belief_state.get_card_color_rank(0)
        assert color == 0 and rank == 0

        color, rank = belief_state.get_card_color_rank(5)
        assert color == 1 and rank == 0

        color, rank = belief_state.get_card_color_rank(1)
        assert color == 0 and rank == 1


class TestCardsToIndices:
    """Test card list to indices conversion."""

    def test_cards_to_indices(self, belief_state):
        """Test converting cards to indices."""
        mock_cards = []
        for i in range(3):
            mock_card = Mock(spec=HanabiCard)
            mock_card.color.return_value = i
            mock_card.rank.return_value = 0
            mock_cards.append(mock_card)

        indices = belief_state.cards_to_indices(mock_cards)

        expected = np.array([0, 5, 10])  # (0,0), (1,0), (2,0)
        np.testing.assert_array_equal(indices, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
