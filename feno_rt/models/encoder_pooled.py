import torch.nn as nn
from .fno_block import FNOBlock
from .self_attn_patch import PatchifiedSelfAttentionBlocks


class EncoderFNOAttentionPooled(nn.Module):
    def __init__(self, input_dim, enc_dim, enc_depth, enc_num_heads,
                 H_vel, W_vel, H_lat, W_lat, patchify, P,
                 coord_dim, pos_embed_dim, modes1, modes2, width, dropout_rate,
                 init_weights="truncnormal"):
        super().__init__()
        self.input_dim = input_dim
        self.enc_dim = enc_dim
        self.H_vel = H_vel
        self.W_vel = W_vel
        self.H_lat = H_lat
        self.W_lat = W_lat

        self.fno = FNOBlock(
            H=H_vel, W=W_vel,
            feat_channels=input_dim,
            out_channels=enc_dim,
            coord_dim=coord_dim,
            pos_embed_dim=pos_embed_dim,
            modes1=modes1, modes2=modes2,
            width=width, dropout_rate=dropout_rate,
        )

        self.patchified_self_attention_blocks = PatchifiedSelfAttentionBlocks(
            P=P, H=H_lat, W=W_lat,
            input_dim=enc_dim, dim=enc_dim,
            num_heads=enc_num_heads,
            enc_depth=enc_depth,
            init_weights=init_weights,
        )

    def forward(self, x):
        x = self.fno(x, query_H=self.H_lat, query_W=self.W_lat)
        x = self.patchified_self_attention_blocks(x)
        return x
