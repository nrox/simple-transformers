import math
import os
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn, Tensor
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator

from src.data import batchify, get_batch, data_process
from src.model import TransformerModel, generate_square_subsequent_mask
from src.training import train


def evaluate(model: nn.Module,
             eval_data: Tensor,
             bptt: int,
             ntokens: int,
             criterion: Any,
             device: Any) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i, bptt)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
