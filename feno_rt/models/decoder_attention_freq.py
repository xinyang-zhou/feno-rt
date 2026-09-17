"""带频率条件的解码器：在交叉注意力后对 query 做 FiLM 调制。

频率 f 以两种形式编码为"源条件向量" c(f):
    1. f 的正弦多频嵌入 (f / f_ref 的多频 sin/cos)
    2. Ricker 子波归一化振幅谱在固定频点上的采样 (对数尺度)

c(f) 经 MLP 生成 gamma/beta, 在 pred 之前对 query 特征做 FiLM:
    query = gamma * query + beta

不与坐标一起输入编码器, 也不作为坐标拼进 sincos 嵌入。
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverBlock

from .pos_emb import ContinuousSincosEmbed
from .self_attn_patch import PatchifiedSelfAttentionBlocks

@dataclass(frozen=True)
class DecoderLayerContext:
    """Frequency/geometry-independent state for one decoder layer."""

    tokens: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class DecoderStaticContext:
    """All medium-only decoder state and projected cross-attention K/V."""

    layers: Tuple[DecoderLayerContext, ...]

    @property
    def device(self):
        return self.layers[0].tokens.device

    @property
    def dtype(self):
        return self.layers[0].tokens.dtype


@dataclass(frozen=True)
class WaveletContext:
    """Wavelet tokens and their projected cross-attention K/V."""

    tokens: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

# 正弦多频嵌入 -> NeRF Fourier Features Spectral Bias
class FreqConditioning(nn.Module):
    def __init__(self, freq_cond_dim=64, K=24, nu_min=1.0, nu_max=100.0,
                 f_ref=40.0, sin_dim=32):
        super().__init__()
        self.K = K
        self.f_ref = f_ref
        self.sin_dim = sin_dim
        nu = torch.exp(torch.linspace(np.log(nu_min), np.log(nu_max), K)) # 对数轴
        self.register_buffer('nu', nu)
        self.proj = nn.Sequential(
            LinearProjection(sin_dim + K, freq_cond_dim * 2, init_weights='truncnormal'),
            nn.GELU(),
            LinearProjection(freq_cond_dim * 2, freq_cond_dim, init_weights='truncnormal'),
        ) # MLP 将频率的信息编码成64维向量

    def forward(self, freq):
        f = freq.view(-1).float()                       # (batch,)
        fn = f / self.f_ref  # self.f_ref -> 归一化
        i = torch.arange(self.sin_dim // 2, device=f.device, dtype=f.dtype) # 生成i=[0, ..., 15]多频正弦嵌入的频率向量
        ang = fn[:, None] * (torch.pi * (2.0 ** i))[None, :] # 相位-> [1, ..., 32768] 逐倍频程增长的频率系数
        sin_emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (batch, sin_dim)
        u = self.nu[None, :] / f[:, None]               # (batch, K)
        A = u ** 2 * torch.exp(1.0 - u ** 2)            # 归一化谱, 峰值=1
        logA = torch.log(A + 1e-6)
        c = torch.cat([sin_emb, logA], dim=-1)          # (batch, sin_dim + K)
        return self.proj(c)                             # (batch, freq_cond_dim)


class SourceWaveletTokens(nn.Module):
    """把源 Ricker 子波编码成 token 序列, 供解码器交叉注意力使用。

    子波由频率 f 唯一决定:
        t = t - peak_time,  peak_time = 1.5/f
        w = (1 - 2*pi^2*f^2*t^2) * exp(-pi^2*f^2*t^2)
    在输出 1024 点时间网格 (0..T) 上采样, 与数据降采样网格等价。
    """
    def __init__(self, dim, nt=1024, T=0.5, init_weights='truncnormal002'):
        super().__init__()
        self.dim = dim
        self.nt = nt
        self.T = T
        self.time_embed = ContinuousSincosEmbed(dim=dim, ndim=1)
        self.val_proj = LinearProjection(1, dim, init_weights=init_weights)

    def forward(self, freq):
        f = freq.view(-1).float()                       # (batch,)
        t_norm = torch.linspace(0.0, 1.0, self.nt, device=f.device, dtype=f.dtype)
        t = t_norm * self.T
        t_norm = t_norm.unsqueeze(-1)                   # (nt, 1) for 1d sincos embed
        dt_ = t[None, :] - (1.5 / f.clamp_min(1e-3))[:, None]
        a = (torch.pi * f[:, None] * dt_) ** 2
        w = (1.0 - 2.0 * a) * torch.exp(-a)             # (batch, nt)
        pos = self.time_embed(t_norm)                   # (nt, dim)
        w_emb = self.val_proj(w[..., None])             # (batch, nt, dim)
        return w_emb + pos[None]                        # (batch, nt, dim)


class SelfCrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, P, H, W, init_weights):
        super().__init__()
        self.self_attn = PatchifiedSelfAttentionBlocks(
            P=P, H=H, W=W, input_dim=dim, dim=dim, num_heads=num_heads,
            enc_depth=1, init_weights=init_weights)
        self.cross_attn = PerceiverBlock(
            kv_dim=dim, dim=dim, num_heads=num_heads, init_weights=init_weights)

    def forward(self, x, query):
        x = self.self_attn(x)
        query = self.cross_attn(q=query, kv=x)
        return x, query


class DecoderAttentionFreq(nn.Module):
    def __init__(self, input_dim, output_dim, dec_dim, dec_depth, dec_num_heads,
                 enforce_reciprocity=True, patchify=True, P=2, H=None, W=None,
                 init_weights='truncnormal002', freq_cond_dim=64,
                 use_wavelet_attn=True, **kwargs):
        super().__init__(**kwargs)
        self.enforce_reciprocity = enforce_reciprocity
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dec_dim = dec_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.init_weights = init_weights
        self.freq_cond_dim = freq_cond_dim
        self.input_proj = LinearProjection(input_dim, dec_dim, init_weights=init_weights, optional=True)
        assert patchify, "patchify must be True"
        assert H is not None and W is not None

        self.blocks = nn.ModuleList([
            SelfCrossAttentionBlock(dim=dec_dim, num_heads=dec_num_heads, P=P, H=H, W=W, init_weights=init_weights)
            for _ in range(dec_depth)
        ])

        self.pos_embed = ContinuousSincosEmbed(dim=dec_dim, ndim=2)
        self.query_proj = nn.Sequential(
            LinearProjection(dec_dim * 2, dec_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(dec_dim * 4, dec_dim, init_weights=init_weights),
        )

        self.freq_cond = FreqConditioning(freq_cond_dim=freq_cond_dim)
        self.film = nn.Sequential(
            LinearProjection(freq_cond_dim, freq_cond_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(freq_cond_dim * 4, dec_dim * 2, init_weights=init_weights),
        )

        self.use_wavelet_attn = use_wavelet_attn
        if use_wavelet_attn:
            self.wavelet_tokens = SourceWaveletTokens(dim=dec_dim, nt=output_dim,
                                                      init_weights=init_weights)
            self.wavelet_attn = PerceiverBlock(dim=dec_dim, num_heads=dec_num_heads,
                                               init_weights=init_weights)

        self.pred = nn.Sequential(
            LinearProjection(dec_dim, output_dim, init_weights=init_weights),
        )

    @staticmethod
    def _project_perceiver_kv(block, tokens):
        """Apply the exact norm/KV projection used by PerceiverBlock."""
        attn = block.attn
        if attn.concat_query_to_kv:
            raise NotImplementedError("cached K/V does not support concat_query_to_kv")
        projected = attn.kv(block.norm1kv(tokens))
        batch_size, sequence_length, _ = projected.shape
        projected = projected.reshape(
            batch_size,
            sequence_length,
            2,
            attn.num_heads,
            attn.head_dim,
        ).permute(2, 0, 3, 1, 4)
        return projected[0], projected[1]

    def _perceiver_from_projected_kv(self, block, query, k, v):
        """Run a PerceiverBlock while reusing preprojected K/V."""
        if block.training:
            raise RuntimeError("cached decoder contexts are inference-only")
        attn = block.attn
        batch_size, query_length, _ = query.shape
        q = attn.q(block.norm1q(query))
        q = q.reshape(
            batch_size,
            query_length,
            attn.num_heads,
            attn.head_dim,
        ).permute(0, 2, 1, 3)

        if k.shape[0] == 1 and batch_size != 1:
            k = k.expand(batch_size, -1, -1, -1)
            v = v.expand(batch_size, -1, -1, -1)
        elif k.shape[0] != batch_size:
            raise ValueError(
                f"cached K/V batch is {k.shape[0]}, but query batch is {batch_size}"
            )

        residual = F.scaled_dot_product_attention(q, k, v)
        residual = residual.permute(0, 2, 1, 3).reshape(batch_size, query_length, -1)
        residual = block.ls1(attn.proj(residual))
        query = query + residual
        return block.drop_path2(query, block._mlp_residual_path)

    def _build_geometry_query(self, src_pos, rec_pos):
        src_pos = self.pos_embed(src_pos)
        rec_pos = self.pos_embed(rec_pos)
        if self.enforce_reciprocity:
            q1 = self.query_proj(torch.cat([src_pos, rec_pos], dim=-1))
            q2 = self.query_proj(torch.cat([rec_pos, src_pos], dim=-1))
            return (q1 + q2) / 2
        return self.query_proj(torch.cat([src_pos, rec_pos], dim=-1))

    def prepare_static_context(self, x):
        """Build medium-only layer tokens and cross-attention K/V once."""
        if self.training:
            raise RuntimeError("decoder context caching is inference-only")
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError(
                "decoder static context expects a single medium latent with shape "
                f"(1, tokens, dim), got {tuple(x.shape)}"
            )
        x = self.input_proj(x)
        layers = []
        for block in self.blocks:
            x = block.self_attn(x)
            k, v = self._project_perceiver_kv(block.cross_attn, x)
            layers.append(DecoderLayerContext(tokens=x, k=k, v=v))
        return DecoderStaticContext(layers=tuple(layers))

    def prepare_geometry_prefix(self, context, src_pos, rec_pos):
        """Run the frequency-independent query path against cached medium K/V."""
        if len(context.layers) != len(self.blocks):
            raise ValueError("decoder context depth does not match the decoder")
        query = self._build_geometry_query(src_pos, rec_pos)
        for block, layer in zip(self.blocks, context.layers):
            query = self._perceiver_from_projected_kv(
                block.cross_attn,
                query,
                layer.k,
                layer.v,
            )
        return query

    def prepare_wavelet_context(self, freq):
        """Build exact-frequency wavelet tokens and projected K/V."""
        if not self.use_wavelet_attn:
            raise RuntimeError("wavelet attention is disabled")
        tokens = self.wavelet_tokens(freq)
        k, v = self._project_perceiver_kv(self.wavelet_attn, tokens)
        return WaveletContext(tokens=tokens, k=k, v=v)

    def forward_from_geometry_prefix(self, query, freq, wavelet_context=None):
        """Finish frequency-dependent decoding from a geometry prefix."""
        c = self.freq_cond(freq)
        if self.use_wavelet_attn:
            if wavelet_context is None:
                query = self.wavelet_attn(q=query, kv=self.wavelet_tokens(freq))
            else:
                query = self._perceiver_from_projected_kv(
                    self.wavelet_attn,
                    query,
                    wavelet_context.k,
                    wavelet_context.v,
                )

        gb = self.film(c)
        gamma = gb[:, None, :self.dec_dim]
        beta = gb[:, None, self.dec_dim:]
        return self.pred(query * gamma + beta)

    def forward_with_static_context(
        self, context, src_pos, rec_pos, freq, wavelet_context=None
    ):
        query = self.prepare_geometry_prefix(context, src_pos, rec_pos)
        return self.forward_from_geometry_prefix(query, freq, wavelet_context)

    def forward(self, x, src_pos, rec_pos, freq):
        x = self.input_proj(x)
        src_pos = self.pos_embed(src_pos)
        rec_pos = self.pos_embed(rec_pos)
        if self.enforce_reciprocity:
            q1 = self.query_proj(torch.cat([src_pos, rec_pos], dim=-1))
            q2 = self.query_proj(torch.cat([rec_pos, src_pos], dim=-1))
            query = (q1 + q2) / 2
        else:
            query = self.query_proj(torch.cat([src_pos, rec_pos], dim=-1))

        c = self.freq_cond(freq)
        for block in self.blocks:
            x, query = block(x, query)

        if self.use_wavelet_attn:
            query = self.wavelet_attn(q=query, kv=self.wavelet_tokens(freq)) # q: [B, 700, 128] kv: [B, 1024, 128]

        gb = self.film(c)        # (batch, dec_dim*2) [B, 256]
        gamma = gb[:, None, :self.dec_dim] # [B, 1, 128]
        beta = gb[:, None, self.dec_dim:] # [B, 1, 128]
        query = query * gamma + beta

        return self.pred(query)
