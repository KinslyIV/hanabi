from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from rl_hanabi.model.tokenizer import TokenizationConfig


@dataclass(frozen=True)
class ActionDecoderConfig:
    num_colors: int
    num_ranks: int
    max_cards: int
    hand_size: int
    num_players: int
    num_heads: int = 4
    num_layers: int = 4
    d_model: int = 128
    dropout: float = 0.2
    bias: bool = False



class SelfAttention(nn.Module):

    def __init__(self, config: ActionDecoderConfig):
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

    def forward(self, x, *, attn_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, embed_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.proj(attn)
    

class RecurrentFeedForward(nn.Module):

    def __init__(self, config: ActionDecoderConfig, n_layers=1):
        super().__init__()
        self.in_proj = nn.Linear(config.d_model, config.d_model * 4)
        self.cell = nn.LSTMCell(config.d_model * 4, config.d_model * 4)
        self.out_proj = nn.Linear(config.d_model * 4, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None


    def reset_state(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        if reset_mask is None:
            self._h = None
            self._c = None
            return

        if self._h is None or self._c is None:
            return

        if reset_mask.dim() != 1 or reset_mask.size(0) != self._h.size(0):
            self._h = None
            self._c = None
            return

        mask = reset_mask.to(self._h.device).view(-1, 1, 1)
        self._h = self._h.masked_fill(mask, 0.0)
        self._c = self._c.masked_fill(mask, 0.0)

    def forward(self, x):
        batch_size, context_len, d_model = x.shape
        x_proj = self.in_proj(x)
        x_flat = x_proj.contiguous().view(batch_size * context_len, d_model * 4)
        if self._h is None or self._c is None:
            h = x.new_zeros(batch_size, context_len, d_model * 4)
            c = x.new_zeros(batch_size, context_len, d_model * 4)
        else:
            prev_batch, prev_len, prev_dim = self._h.shape
            if prev_dim != d_model * 4 or prev_batch != batch_size:
                h = x.new_zeros(batch_size, context_len, d_model * 4)
                c = x.new_zeros(batch_size, context_len, d_model * 4)
            elif context_len == prev_len:
                h = self._h
                c = self._c
            elif context_len == prev_len + 1:
                pad = x.new_zeros(batch_size, 1, d_model * 4)
                h = torch.cat([self._h, pad], dim=1)
                c = torch.cat([self._c, pad], dim=1)
            else:
                h = x.new_zeros(batch_size, context_len, d_model * 4)
                c = x.new_zeros(batch_size, context_len, d_model * 4)

        h_flat = h.contiguous().view(batch_size * context_len, d_model * 4)
        c_flat = c.contiguous().view(batch_size * context_len, d_model * 4)
        h_flat, c_flat = self.cell(x_flat, (h_flat, c_flat))
        self._h = h_flat.view(batch_size, context_len, d_model * 4).detach()
        self._c = c_flat.view(batch_size, context_len, d_model * 4).detach()
        out = h_flat.view(batch_size, context_len, d_model * 4)
        return self.dropout(self.out_proj(out))
    

class Block(nn.Module):

    def __init__(self, config : ActionDecoderConfig, n_ff_layers=1):
        super().__init__()

        self.multi_head = SelfAttention(config)
        self.ffwd = RecurrentFeedForward(config, n_layers=n_ff_layers)
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn_residual = nn.GRUCell(config.d_model, config.d_model)
        self.ff_residual = nn.GRUCell(config.d_model, config.d_model)


    def _gru_residual(self, cell: nn.GRUCell, x: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.contiguous().view(batch_size * seq_len, d_model)
        update_flat = update.contiguous().view(batch_size * seq_len, d_model)
        out_flat = cell(update_flat, x_flat)
        return out_flat.view(batch_size, seq_len, d_model)

    def reset_state(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        self.ffwd.reset_state(reset_mask)

    def forward(self, x, *, attn_mask: Optional[torch.Tensor] = None):
        attn_out = self.multi_head(self.ln1(x), attn_mask=attn_mask)
        x = self._gru_residual(self.attn_residual, x, attn_out)
        ff_out = self.ffwd(self.ln2(x))
        x = self._gru_residual(self.ff_residual, x, ff_out)
        return x

class ActionDecoder(nn.Module):

    def __init__(
        self,
        config: Optional[ActionDecoderConfig] = None,
        *,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        d_model: Optional[int] = None,
        token_config: TokenizationConfig,
    ):

        super().__init__()
        
        if config is None:
            if num_heads is None or num_layers is None or d_model is None or token_config is None:
                raise ValueError("Provide either config or all max_* parameters")
            config = ActionDecoderConfig(
                num_colors=token_config.num_colors,
                num_ranks=token_config.num_ranks,
                max_cards=token_config.total_cards,
                hand_size=token_config.hand_size,
                num_players=token_config.num_players,
                num_heads=num_heads,
                num_layers=num_layers,
                d_model=d_model
            )

        # Store max dimensions as instance attributes
        self.num_colors = config.num_colors
        self.num_ranks = config.num_ranks
        self.hand_size = config.hand_size
        self.num_players = config.num_players
        self.config = config

        # We prepend a learned state token to the embedded sequence, so all
        # positions after it are shifted by +1 compared to the raw token IDs.
        self.hand_start = 1 + 4 + self.num_colors
        self.hand_len = self.num_players * self.hand_size

        self.pad_token = token_config.pad_token

        self.action_space_size = token_config.action_space_size + 1
        self.total_card_tokens = token_config.total_card_tokens
        self.context_size = token_config.context_size

        self.card_emb = nn.Embedding(self.total_card_tokens, config.d_model, padding_idx=self.pad_token)
        self.player_emb = nn.Embedding(self.num_players, config.d_model)
        self.action_emb = nn.Embedding(self.action_space_size, config.d_model, padding_idx=self.pad_token)
        self.life_proj = nn.Linear(1, config.d_model)
        self.info_proj = nn.Linear(1, config.d_model)

        # Learned state token (prepended at the embedding level; not part of token IDs)
        self.state_token = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # +1 to account for the prepended state token embedding
        self.pos_emb = nn.Embedding(self.context_size + 1, config.d_model)

        self.card_action_head = nn.Linear(config.d_model, 4)
        self.state_value_head = nn.Linear(config.d_model, 1)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)



    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dtype != torch.long:
            x = x.long()

        if x.size(1) < 4:
            raise ValueError("Expected at least 4 tokens: current_player, life, info, action")
        if x.size(1) > self.context_size:
            raise ValueError("Sequence length exceeds context_size")
        if x.size(1) < self.context_size:
            pad_len = self.context_size - x.size(1)
            x = F.pad(x, (0, pad_len), value=self.pad_token)

        token_ids = x

        batch_size = x.size(0)

        current_player_tokens = x[:, 0]
        life_tokens = x[:, 1].float().unsqueeze(-1)
        info_tokens = x[:, 2].float().unsqueeze(-1)
        action_tokens = x[:, 3]
        card_tokens = x[:, 4:]

        current_player_emb = self.player_emb(current_player_tokens).unsqueeze(1)
        life_emb = self.life_proj(life_tokens).unsqueeze(1)
        info_emb = self.info_proj(info_tokens).unsqueeze(1)
        action_emb = self.action_emb(action_tokens).unsqueeze(1)
        card_emb = self.card_emb(card_tokens)

        state_emb = self.state_token.expand(batch_size, -1, -1)
        x = torch.cat([state_emb, current_player_emb, life_emb, info_emb, action_emb, card_emb], dim=1)

        key_padding = token_ids == self.pad_token
        key_padding = torch.cat(
            [torch.zeros(batch_size, 1, device=token_ids.device, dtype=torch.bool), key_padding],
            dim=1,
        )
        attn_mask = key_padding.unsqueeze(1).unsqueeze(2)

        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        x = self.ln(x)

        hand_end = min(self.hand_start + self.hand_len, x.size(1))
        if self.hand_start >= hand_end:
            raise ValueError("Hand token range is empty for action head")

        hand_hidden = x[:, self.hand_start:hand_end, :]
        card_action_logits = self.card_action_head(hand_hidden)
        value = self.state_value_head(x[:, 0, :])

        return card_action_logits, value

    def reset_state(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        for block in self.blocks:
            block.reset_state(reset_mask)

