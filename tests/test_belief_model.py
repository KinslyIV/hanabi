"""
Tests for the ActionDecoder belief model.
"""

import pytest
import torch
import numpy as np
from rl_hanabi.model.belief_model import ActionDecoder


class TestActionDecoderInitialization:
    """Test ActionDecoder initialization and architecture."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = ActionDecoder(
            max_num_colors=5,
            max_num_ranks=5,
            max_hand_size=5,
            max_num_players=2,
        )
        assert model is not None
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'color_head')
        assert hasattr(model, 'rank_head')

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = ActionDecoder(
            max_num_colors=4,
            max_num_ranks=6,
            max_hand_size=8,
            max_num_players=3,
            num_heads=8,
            num_layers=6,
            d_model=256,
            action_dim=4,
        )
        assert model is not None

    def test_device_placement(self):
        """Test that model can be moved to different devices."""
        model = ActionDecoder(5, 5, 5, 2)
        model_cpu = model.cpu()
        assert next(model_cpu.parameters()).device.type == 'cpu'

        if torch.cuda.is_available():
            model_cuda = model.cuda()
            assert next(model_cuda.parameters()).device.type == 'cuda'


class TestActionDecoderForward:
    """Test ActionDecoder forward pass."""

    @pytest.fixture
    def model(self):
        """Create a default ActionDecoder model for tests."""
        return ActionDecoder(
            max_num_colors=5,
            max_num_ranks=5,
            max_hand_size=5,
            max_num_players=2,
        )

    def test_forward_pass_basic(self, model):
        """Test basic forward pass with valid inputs."""
        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.randint(0, 2, (batch_size, num_players, hand_size)).float()
        move_target_player = torch.tensor([0, 1], dtype=torch.long)
        acting_player = torch.tensor([0, 1], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)
        assert rank_logits.shape == (batch_size, hand_size, num_ranks)

    def test_output_shapes(self, model):
        """Test that output shapes are correct for various batch sizes."""
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        for batch_size in [1, 4, 8]:
            slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
            affected_mask = torch.randint(0, 2, (batch_size, num_players, hand_size)).float()
            move_target_player = torch.randint(0, num_players, (batch_size,), dtype=torch.long)
            acting_player = torch.randint(0, num_players, (batch_size,), dtype=torch.long)
            action = torch.randn(batch_size, 4)
            fireworks = torch.randn(batch_size, num_colors)
            discard_pile = torch.randn(batch_size, num_colors * num_ranks)

            color_logits, rank_logits = model(
                slot_beliefs=slot_beliefs,
                affected_mask=affected_mask,
                move_target_player=move_target_player,
                acting_player=acting_player,
                action=action,
                fireworks=fireworks,
                discard_pile=discard_pile,
            )

            assert color_logits.shape == (batch_size, hand_size, num_colors)
            assert rank_logits.shape == (batch_size, hand_size, num_ranks)

    def test_forward_with_zero_inputs(self, model):
        """Test forward pass with zero inputs."""
        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.zeros(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.zeros(batch_size, dtype=torch.long)
        acting_player = torch.zeros(batch_size, dtype=torch.long)
        action = torch.zeros(batch_size, 4)
        fireworks = torch.zeros(batch_size, num_colors)
        discard_pile = torch.zeros(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert not torch.isnan(color_logits).any()
        assert not torch.isnan(rank_logits).any()

    def test_forward_with_ones_inputs(self, model):
        """Test forward pass with ones inputs."""
        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.ones(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.ones(batch_size, num_players, hand_size)
        move_target_player = torch.ones(batch_size, dtype=torch.long)
        acting_player = torch.ones(batch_size, dtype=torch.long)
        action = torch.ones(batch_size, 4)
        fireworks = torch.ones(batch_size, num_colors)
        discard_pile = torch.ones(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert not torch.isnan(color_logits).any()
        assert not torch.isnan(rank_logits).any()


class TestActionDecoderGradients:
    """Test gradient flow through ActionDecoder."""

    def test_gradients_flow(self):
        """Test that gradients flow through the model."""
        model = ActionDecoder(5, 5, 5, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks,
                                   requires_grad=False)
        affected_mask = torch.randint(0, 2, (batch_size, num_players, hand_size)).float()
        move_target_player = torch.tensor([0, 1], dtype=torch.long)
        acting_player = torch.tensor([0, 1], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        # Create dummy targets and loss
        color_targets = torch.randint(0, num_colors, (batch_size, hand_size))
        rank_targets = torch.randint(0, num_ranks, (batch_size, hand_size))

        color_loss = torch.nn.functional.cross_entropy(
            color_logits.view(-1, num_colors),
            color_targets.view(-1)
        )
        rank_loss = torch.nn.functional.cross_entropy(
            rank_logits.view(-1, num_ranks),
            rank_targets.view(-1)
        )

        loss = color_loss + rank_loss
        loss.backward()

        # Check that gradients exist and are non-zero for some parameters
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        assert has_gradients, "Gradients should flow through the model"

    def test_optimizer_step(self):
        """Test that optimizer can perform a step."""
        model = ActionDecoder(5, 5, 5, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        initial_params = [p.clone() for p in model.parameters()]

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.randint(0, 2, (batch_size, num_players, hand_size)).float()
        move_target_player = torch.tensor([0, 1], dtype=torch.long)
        acting_player = torch.tensor([0, 1], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        color_targets = torch.randint(0, num_colors, (batch_size, hand_size))
        rank_targets = torch.randint(0, num_ranks, (batch_size, hand_size))

        color_loss = torch.nn.functional.cross_entropy(
            color_logits.view(-1, num_colors),
            color_targets.view(-1)
        )
        rank_loss = torch.nn.functional.cross_entropy(
            rank_logits.view(-1, num_ranks),
            rank_targets.view(-1)
        )

        loss = color_loss + rank_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that parameters have changed
        params_changed = False
        for initial_param, current_param in zip(initial_params, model.parameters()):
            if not torch.allclose(initial_param, current_param):
                params_changed = True
                break

        assert params_changed, "Parameters should be updated by optimizer"


class TestActionDecoderDifferentArchitectures:
    """Test ActionDecoder with different architectural configurations."""

    def test_single_head_attention(self):
        """Test with single attention head."""
        model = ActionDecoder(
            max_num_colors=5,
            max_num_ranks=5,
            max_hand_size=5,
            max_num_players=2,
            num_heads=1,
            num_layers=1,
            d_model=64,
        )

        batch_size = 1
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.tensor([0], dtype=torch.long)
        acting_player = torch.tensor([0], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)
        assert rank_logits.shape == (batch_size, hand_size, num_ranks)

    def test_deep_model(self):
        """Test with deep architecture."""
        model = ActionDecoder(
            max_num_colors=5,
            max_num_ranks=5,
            max_hand_size=5,
            max_num_players=2,
            num_heads=8,
            num_layers=12,
            d_model=256,
        )

        batch_size = 1
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 2

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.tensor([0], dtype=torch.long)
        acting_player = torch.tensor([0], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)
        assert rank_logits.shape == (batch_size, hand_size, num_ranks)


class TestActionDecoderEdgeCases:
    """Test ActionDecoder with edge cases."""

    def test_single_player(self):
        """Test with single player (edge case)."""
        model = ActionDecoder(5, 5, 5, 1)

        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 1

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.zeros(batch_size, dtype=torch.long)
        acting_player = torch.zeros(batch_size, dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)

    def test_large_hand_size(self):
        """Test with large hand size."""
        model = ActionDecoder(5, 5, 20, 2)

        batch_size = 1
        num_colors = 5
        num_ranks = 5
        hand_size = 20
        num_players = 2

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.tensor([0], dtype=torch.long)
        acting_player = torch.tensor([0], dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)

    def test_many_players(self):
        """Test with many players."""
        model = ActionDecoder(5, 5, 5, 5)

        batch_size = 2
        num_colors = 5
        num_ranks = 5
        hand_size = 5
        num_players = 5

        slot_beliefs = torch.randn(batch_size, num_players, hand_size, num_colors + num_ranks)
        affected_mask = torch.zeros(batch_size, num_players, hand_size)
        move_target_player = torch.randint(0, num_players, (batch_size,), dtype=torch.long)
        acting_player = torch.randint(0, num_players, (batch_size,), dtype=torch.long)
        action = torch.randn(batch_size, 4)
        fireworks = torch.randn(batch_size, num_colors)
        discard_pile = torch.randn(batch_size, num_colors * num_ranks)

        color_logits, rank_logits = model(
            slot_beliefs=slot_beliefs,
            affected_mask=affected_mask,
            move_target_player=move_target_player,
            acting_player=acting_player,
            action=action,
            fireworks=fireworks,
            discard_pile=discard_pile,
        )

        assert color_logits.shape == (batch_size, hand_size, num_colors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
