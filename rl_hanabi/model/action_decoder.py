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
    action_dim: int = 4
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

    def forward(self, x):
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
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.proj(attn)
    

class FeedForward(nn.Module):

    def __init__(self, config: ActionDecoderConfig, n_layers=1, activation=nn.Module):
        super().__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.net = nn.Sequential()

        self.net.append(nn.Linear(config.d_model, config.d_model * 4))
        self.net.append(self.activation())

        for _ in range(self.n_layers - 1):
            self.net.append(nn.Linear(config.d_model*4, config.d_model*4))
            self.net.append(self.activation())

        # Adding Projection
        self.net.append(nn.Linear(config.d_model*4, config.d_model))

        self.net.append(nn.Dropout(config.dropout))

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, config : ActionDecoderConfig, n_ff_layers=1):
        super().__init__()

        self.multi_head = SelfAttention(config)
        self.ffwd = FeedForward(config, n_layers=n_ff_layers, activation=nn.GELU)
        self.ln1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.d_model, bias=config.bias)
        

    def forward(self, x):
        x = x + self.multi_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class ActionDecoder(nn.Module):

    def __init__(
        self,
        config: Optional[ActionDecoderConfig] = None,
        *,
        num_colors: Optional[int] = None,
        num_ranks: Optional[int] = None,
        max_cards: Optional[int] = None,
        hand_size: Optional[int] = None,
        num_players: Optional[int] = None,
        num_heads: int = 4,
        num_layers: int = 4,
        d_model: int = 128,
        action_dim: int = 4,
        token_config: TokenizationConfig,
    ):

        super().__init__()
        
        if config is None:
            if num_colors is None or num_ranks is None or hand_size is None or num_players is None or max_cards is None:
                raise ValueError("Provide either config or all max_* parameters")
            config = ActionDecoderConfig(
                num_colors=num_colors,
                num_ranks=num_ranks,
                max_cards=max_cards,
                hand_size=hand_size,
                num_players=num_players,
                num_heads=num_heads,
                num_layers=num_layers,
                d_model=d_model,
                action_dim=action_dim,
            )

        # Store max dimensions as instance attributes
        self.num_colors = config.num_colors
        self.num_ranks = config.num_ranks
        self.hand_size = config.hand_size
        self.num_players = config.num_players
        self.config = config

        self.hand_start = 3 + self.num_colors
        self.hand_len = self.num_players * self.hand_size

        self.pad_token = token_config.pad_token

        self.action_space_size = token_config.action_space_size
        self.total_card_tokens = token_config.total_card_tokens
        self.context_size = token_config.context_size

        self.card_emb = nn.Embedding(self.total_card_tokens, config.d_model)
        self.action_emb = nn.Embedding(self.action_space_size, config.d_model)
        self.life_proj = nn.Linear(1, config.d_model)
        self.info_proj = nn.Linear(1, config.d_model)
        self.pos_emb = nn.Embedding(self.context_size, config.d_model)

        self.card_action_head = nn.Linear(config.d_model, 4)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.d_model, bias=config.bias)

        self._prev_tokens = None
        self._prev_out = None


    def reset_cache(self) -> None:
        self._prev_tokens = None
        self._prev_out = None


    def forward(self, x: torch.Tensor, *, use_cache: bool = True, detach_cache: bool = True) -> torch.Tensor:
        if x.dtype != torch.long:
            x = x.long()

        if x.size(1) < 3:
            raise ValueError("Expected at least 3 tokens: life, info, action")
        if x.size(1) > self.context_size:
            raise ValueError("Sequence length exceeds context_size")
        if x.size(1) < self.context_size:
            pad_len = self.context_size - x.size(1)
            x = F.pad(x, (0, pad_len), value=self.pad_token)

        token_ids = x

        life_tokens = x[:, 0].float().unsqueeze(-1)
        info_tokens = x[:, 1].float().unsqueeze(-1)
        action_tokens = x[:, 2]
        card_tokens = x[:, 3:]

        life_emb = self.life_proj(life_tokens).unsqueeze(1)
        info_emb = self.info_proj(info_tokens).unsqueeze(1)
        action_emb = self.action_emb(action_tokens).unsqueeze(1)
        card_emb = self.card_emb(card_tokens)

        x = torch.cat([life_emb, info_emb, action_emb, card_emb], dim=1)

        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos)

        if use_cache and self._prev_out is not None and self._prev_tokens is not None:
            same_mask = (self._prev_tokens == token_ids)
            same_mask = same_mask & (token_ids != self.pad_token)
            cache = self._prev_out.detach() if detach_cache else self._prev_out
            x = x + cache * same_mask.unsqueeze(-1)

        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if use_cache:
            self._prev_tokens = token_ids.detach()
            self._prev_out = x

        
        hand_end = min(self.hand_start + self.hand_len, x.size(1))
        if self.hand_start >= hand_end:
            raise ValueError("Hand token range is empty for action head")

        hand_hidden = x[:, self.hand_start:hand_end, :]
        card_action_logits = self.card_action_head(hand_hidden)
        return card_action_logits