import torch.nn as nn
from .encoder_pooled import EncoderFNOAttentionPooled
from .decoder_attention_freq import DecoderAttentionFreq


class FENOFreq(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = EncoderFNOAttentionPooled(**encoder_config)
        decoder_config['input_dim'] = encoder_config['enc_dim']
        self.decoder = DecoderAttentionFreq(**decoder_config)

    def forward(self, x, src_pos, rec_pos, freq):
        x = self.encoder(x)
        x = self.decoder(x, src_pos, rec_pos, freq)
        return x
