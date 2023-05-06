import torch
from torch import nn


def test_1():
    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out)


def test_2():
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    src = torch.rand(10, 32, 512)
    out = transformer_encoder(src)
    print(out)


def test_3():
    decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(10, 32, 512)
    tgt = torch.rand(20, 32, 512)
    out = transformer_decoder(tgt, memory)
    print(out)
