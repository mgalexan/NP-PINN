from ML.data_processing import ConcData
from ML.model import BackwardPINN, MLParams

import torch as t
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np

import wandb
import datetime
from tqdm import tqdm

def train_model(model: BackwardPINN, p: MLParams, train_loader: t.utils.data.DataLoader):

    

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print("Training on device", device)

    model = model.to(device)

    # Select the loss function and optimizer
    if p["loss"] == "MSE":
        loss_fn = MSELoss()

    else:
        raise NotImplementedError(f"Error: unsupported loss " + p["loss"])
    
    if p["opt"] == "Adam":
        opt = optim.Adam(model.parameters(), p["lr"])
    
    elif p["opt"] == "SGD":
        opt = optim.SGD(model.parameters(), p["lr"], p["momentum"])
    
    else:
        raise NotImplementedError("Error: unsupported optimizer " + p["opt"])
    

    # Prepare wandb run
    run = wandb.init(config= p, project= "NP-PINN")

    num_epochs = p["num_epochs"]

    # Main training loop
    print("Begin training at " + str(datetime.datetime.now()))
    
    model.train()

    for epoch in tqdm(range(num_epochs)):

        total_loss = 0.0

        for batch in train_loader:

            coords, concs = batch

            train_out = model(coords)

            batch_loss = loss_fn(concs, train_out)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            total_loss += batch_loss.item()
            log_loss = np.log10(total_loss)

        run.log({
            "total_loss" : total_loss,
            "log_loss" : log_loss
        })







