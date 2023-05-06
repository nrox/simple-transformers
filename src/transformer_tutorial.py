# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py

import math
import os
import time
from tempfile import TemporaryDirectory

import torch
from torch import nn

from src.data import DataPreprocessor
from src.evaluation import evaluate
from src.model import TransformerModel
from src.training import train

bptt = 35

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 20
eval_batch_size = 10

emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate

data_wrapper = DataPreprocessor(batch_size=batch_size, eval_batch_size=eval_batch_size, device=device)

train_data = data_wrapper.train_data
val_data = data_wrapper.val_data
test_data = data_wrapper.test_data
ntokens = data_wrapper.ntokens

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

#model = torch.compile(model)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

######################################################################
# Loop over epochs. Save the model if the validation loss is the best
# we've seen so far. Adjust the learning rate after each epoch.

best_val_loss = float('inf')
epochs = 3

with TemporaryDirectory() as tempdir:
    best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()

        train(model, train_data, ntokens, bptt, criterion, optimizer, scheduler, epoch, device)

        val_loss = evaluate(model, val_data, bptt, ntokens, criterion, device)

        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_params_path)

        scheduler.step()
    model.load_state_dict(torch.load(best_model_params_path))  # load best model states

######################################################################
# Evaluate the best model on the test dataset
# -------------------------------------------
#

test_loss = evaluate(model, test_data, bptt, ntokens, criterion, device)

test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {test_ppl:8.2f}')
print('=' * 89)
