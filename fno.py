# Required libraries



"""
Inspired by github: https://github.com/neuraloperator 
Training a FNO on the North sea data

1. loading and preprocessing the data
2. creating FNO model architecture
3. Setting up training components (optimizer, scheduler, losses)
4. Training the model
5. Evaluating predictions



"""
#Import dependencies
import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import numpy as np
import os
from neuralop.models import FNO
from neuralop.utils import count_model_params
from neuralop.training import AdamW
from neuralop import LpLoss, H1Loss
import sys
from neuralop import Trainer

device = "cpu"
#Loading the dataset




operator = FNO(
    n_modes=(64, 64),
    hidden_channels=64,
    in_channels=2,
    out_channels=1,
    projection_channel_ratio=2,
)
operator = operator.to(device)

# Count and display the number of parameters
n_params = count_model_params(operator)
print(f"\nOur model has {n_params} parameters.")
sys.stdout.flush()


optimizer = AdamW(operator.parameters(), lr=1e-2, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)



l2loss = LpLoss(d=2, p=2)  # L2 loss for function values
h1loss = H1Loss(d=2)  # H1 loss includes gradient information

train_loss = h1loss
eval_losses = {"h1": h1loss, "l2": l2loss}

#Training the model
#Display training configuration
print("\n### MODEL ###\n", operator)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()


trainer = Trainer(
    model=operator,
    n_epochs=15,
    device=device,
    data_processor=data_processor,
    wandb_log=False,  # Disable Weights & Biases logging for this tutorial
    eval_interval=5,  # Evaluate every 5 epochs
    use_distributed=False,  # Single GPU/CPU training
    verbose=True,  # Print training progress
)

# Train the model on our Nordic sea dataset. The trainer will:
# 1. Run the forward pass through the FNO
# 2. Compute the H1 loss
# 3. Backpropagate and update weights
# 4. Evaluate on test data every 3 epochs
trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)