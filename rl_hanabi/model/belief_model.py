import torch
from torch import nn

class ActionDecoder(nn.Module):

    def __init__(self, 
                 max_num_colors: int, 
                 max_num_ranks: int, 
                 max_hand_size: int, 
                 max_num_players: int,
                 num_heads: int = 4,
                 num_layers: int = 4,
                 d_model: int = 128,
                 action_dim: int = 4):

        super().__init__()
        
        # Store max dimensions as instance attributes
        self.max_num_colors = max_num_colors
        self.max_num_ranks = max_num_ranks
        self.max_hand_size = max_hand_size
        self.max_num_players = max_num_players

        self.slot_belief_proj = nn.Linear(max_num_colors + max_num_ranks, d_model)
        self.act_proj  = nn.Linear(action_dim, d_model)
        self.firework_proj = nn.Linear(max_num_colors, d_model)
        self.discard_pile_proj = nn.Linear(max_num_colors * max_num_ranks, d_model)

        self.slot_emb  = nn.Embedding(max_hand_size, d_model)
        self.player_emb = nn.Embedding(max_num_players, d_model)
        self.move_target_player_emb = nn.Embedding(1, d_model) 
        self.affected_emb = nn.Embedding(2, d_model) # 0: unaffected, 1: affected

        self.state_token = nn.Parameter(torch.randn(1, 1, d_model))


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers
        )
        
        self.color_head = nn.Linear(d_model, max_num_colors)
        self.rank_head  = nn.Linear(d_model, max_num_ranks)

        action_space_size = 2 * max_hand_size + (max_num_players - 1) * max_num_colors + (max_num_players - 1) * max_num_ranks

        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_space_size),
        )


    def forward(self,
                slot_beliefs,     # [B, P, H, C+R]
                affected_mask,    # [B, P, H]     (0/1)
                move_target_player, # [B]
                acting_player,    # [B]
                action,           # [B, action_dim]
                fireworks,        # [B, C]
                discard_pile,     # [B, C*R]
            ):
        device = slot_beliefs.device
        B, P, H, _ = slot_beliefs.shape

        # --------------------------------------------------
        # 1. Flatten player × slot into slot tokens
        # --------------------------------------------------
        slot_beliefs = slot_beliefs.view(B, P * H, -1)      # [B, P*H, C+R]
        x = self.slot_belief_proj(slot_beliefs)             # [B, P*H, d_model]

        # --------------------------------------------------
        # 2. Slot index embedding (position in hand)
        # --------------------------------------------------
        slot_ids = torch.arange(H, device=device).repeat(P)         # [P*H]
        slot_ids = slot_ids.unsqueeze(0).expand(B, P * H)           # [B, P*H]
        x = x + self.slot_emb(slot_ids)

        # --------------------------------------------------
        # 3. Player ownership embedding (who owns this slot)
        # --------------------------------------------------
        player_ids = torch.arange(P, device=device).repeat_interleave(H)
        player_ids = player_ids.unsqueeze(0).expand(B, P * H)
        x = x + self.player_emb(player_ids)

        # --------------------------------------------------
        # 4. Move target embedding (all slots of the move-target hand)
        # --------------------------------------------------
        # compute hand indices of the move target
        hand_start = move_target_player * H
        hand_end   = hand_start + H


        slot_indices = torch.arange(P*H, device=move_target_player.device).unsqueeze(0)  # [1, P*H]
        # broadcasted to [B, P*H] automatically
        # compare with start/end to make mask
        move_target_mask = (slot_indices >= hand_start.unsqueeze(1)) & (slot_indices < hand_end.unsqueeze(1))
        mask = move_target_mask.unsqueeze(-1).float()  # [B, P*H, 1]
        x = x + self.move_target_player_emb(torch.zeros(1, dtype=torch.long, device=device)) * mask   


        # --------------------------------------------------
        # 5. Affected-slot embedding (only slots affected by the move)
        # --------------------------------------------------
        affected = affected_mask.view(B, P*H).long()  # 1 where slot is affected
        x = x + self.affected_emb(affected)  # broadcasted only to affected slots

        # --------------------------------------------------
        # 6. Global conditioning (acting player + action)
        # --------------------------------------------------
        global_tokens = torch.stack([
            self.player_emb(acting_player),
            self.act_proj(action),
            self.firework_proj(fireworks),
            self.discard_pile_proj(discard_pile)], dim=1)   # [B, 4, d_model]

        x = torch.cat([x, global_tokens], dim=1)            # [B, P*H + 4, d_model]
        state_token = self.state_token.expand(B, -1, -1)
        x = torch.cat([x, state_token], dim=1)              # [B, P*H + 5, d_model]

        # --------------------------------------------------
        # 7. Transformer
        # --------------------------------------------------
        x = self.transformer(x)                              # [B, P*H + 5, d_model]

        # --------------------------------------------------
        # 8. Decode first hand's slots
        # --------------------------------------------------
        hand_index = 0 # the hand index to be predicted is always the first
        start = hand_index * H
        end   = start + H
        hand_repr = x[:, start:end, :]

        color_logits = self.color_head(hand_repr)          # [B, num_colors]
        rank_logits  = self.rank_head(hand_repr)           # [B, num_ranks]

        global_repr = x[:, -1]
        action_logits = self.action_head(global_repr)     # [B, action_space_size]

        return color_logits, rank_logits, action_logits