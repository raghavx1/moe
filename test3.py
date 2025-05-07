# Sparsely-Gated Mixture-of-Experts Layers for Regression
# Based on "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538

from moe41 import MoE
import torch
from torch import nn
from torch.optim import Adam

def train(x, y, model, loss_fn, optim):
    # model returns the prediction and the load balancing loss
    y_hat, aux_loss = model(x.float())
    # calculate prediction loss - MSE for regression
    loss = loss_fn(y_hat, y)
    # combine losses
    total_loss = loss + aux_loss
    optim.zero_grad()
    total_loss.backward()
    optim.step()

    print("Training Results - loss: {:.4f}, aux_loss: {:.4f}".format(loss.item(), aux_loss.item()))
    return model

def eval(x, y, model, loss_fn):
    model.eval()
    # model returns the prediction and the load balancing loss
    y_hat, aux_loss = model(x.float(), train=False)
    loss = loss_fn(y_hat, y)
    total_loss = loss + aux_loss
    print("Evaluation Results - loss: {:.4f}, aux_loss: {:.4f}".format(loss.item(), aux_loss.item()))

def dummy_data(batch_size, input_size, output_size):
    # dummy input
    x = torch.rand(batch_size, input_size)
    # dummy target - continuous values for regression
    y = torch.rand(batch_size, output_size)
    return x, y

# arguments
input_size = 100
output_size = 1  # single value regression
num_experts = 10
hidden_size = 64
batch_size = 5
k = 4

# instantiate the MoE layer
model = MoE(input_size, output_size, num_experts, hidden_size, k=k, noisy_gating=True)

# ===== IMPORTANT MODIFICATION =====
# You also need to modify the SparseDispatcher.combine() method in moe.py
# Change the return line from:
# return combined.log()
# to:
# return combined
# This removes the log operation which is not needed for regression

# Use MSE loss for regression
loss_fn = nn.MSELoss()
optim = Adam(model.parameters())

x, y = dummy_data(batch_size, input_size, output_size)

# train
model = train(x, y, model, loss_fn, optim)

# evaluate
x, y = dummy_data(batch_size, input_size, output_size)
eval(x, y, model, loss_fn)