from typing import Any

import torch
from torch import nn, Tensor

from src.transformer_tutorial.data import get_batch
from src.transformer_tutorial.model import generate_square_subsequent_mask


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
