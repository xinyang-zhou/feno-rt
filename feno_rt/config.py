"""Typed configuration for the bundled inference adapter."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class FENOModelConfig:
    original_size: int = 7001
    velocity_height: int = 700
    velocity_width: int = 700
    latent_height: int = 70
    latent_width: int = 70
    output_steps: int = 1024
    receiver_depth: int = 23
    num_receivers: int = 700

    encoder_dim: int = 128
    encoder_depth: int = 6
    encoder_heads: int = 4
    decoder_dim: int = 128
    decoder_depth: int = 4
    decoder_heads: int = 4
    fno_modes1: int = 24
    fno_modes2: int = 24
    fno_width: int = 128
    patch_size: int = 2
    position_embedding_dim: int = 32
    frequency_condition_dim: int = 64
    dropout_rate: float = 0.1
    enforce_reciprocity: bool = True
    use_wavelet_attention: bool = True

    @property
    def domain_extent(self) -> float:
        if self.velocity_height != self.velocity_width:
            raise ValueError("The current coordinate normalization expects a square grid")
        return float(self.velocity_height - 1)

    def encoder_config(self) -> Dict[str, Any]:
        return {
            "input_dim": 1,
            "enc_dim": self.encoder_dim,
            "enc_depth": self.encoder_depth,
            "enc_num_heads": self.encoder_heads,
            "patchify": True,
            "P": self.patch_size,
            "H_vel": self.velocity_height,
            "W_vel": self.velocity_width,
            "H_lat": self.latent_height,
            "W_lat": self.latent_width,
            "modes1": self.fno_modes1,
            "modes2": self.fno_modes2,
            "width": self.fno_width,
            "coord_dim": 2,
            "pos_embed_dim": self.position_embedding_dim,
            "dropout_rate": self.dropout_rate,
            "init_weights": "truncnormal",
        }

    def decoder_config(self) -> Dict[str, Any]:
        return {
            "input_dim": self.encoder_dim,
            "output_dim": self.output_steps,
            "dec_dim": self.decoder_dim,
            "dec_depth": self.decoder_depth,
            "dec_num_heads": self.decoder_heads,
            "enforce_reciprocity": self.enforce_reciprocity,
            "patchify": True,
            "P": self.patch_size,
            "H": self.latent_height,
            "W": self.latent_width,
            "init_weights": "truncnormal002",
            "freq_cond_dim": self.frequency_condition_dim,
            "use_wavelet_attn": self.use_wavelet_attention,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_INFERENCE_CONFIG = FENOModelConfig()
