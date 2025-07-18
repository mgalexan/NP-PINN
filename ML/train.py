from ML.data_processing import ConcData
from ML.model import ForwardPINN, MLParams
from Physics.physloss import pressure_phys_loss

import torch as t
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np

import wandb
import datetime
from tqdm import tqdm

def train_model(model: ForwardPINN, p: MLParams, train_loader: t.utils.data.DataLoader, use_wandb: bool = False):

    

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print("Training on device", device)

    model = model.to(device)

    # Select the loss function and optimizer
    if p["loss"] == "MSE":
        data_loss_fn = MSELoss()
        def loss_fn(data, train_out):
            total_loss = data_loss_fn(data, train_out)

            total_loss_item = total_loss.item()
            total_log_loss = np.log10(total_loss_item)
            loss_dict = {
                "total_loss" : total_loss_item,
                "log_loss" : total_log_loss,
            }

            return total_loss, loss_dict

    elif p["loss"] == "Pressure_Loss":
        data_loss_fn = MSELoss()
        def loss_fn(data, train_out):
            data_loss = data_loss_fn(data, train_out)
            phys_loss =  p["phys_weight"] * pressure_phys_loss(model, model.coloc, model.env.torch_funcs)
            total_loss = data_loss + phys_loss

            total_loss_item = total_loss.item()
            total_log_loss = np.log10(total_loss_item)
            data_loss_item = data_loss.item()
            phys_loss_item = phys_loss.item()
            loss_dict = {
                "total_loss" : total_loss_item,
                "log_loss" : total_log_loss,
                "data_loss" : data_loss_item,
                "physics_loss" : phys_loss_item
            }

            return total_loss, loss_dict

            
            


    else:
        raise NotImplementedError(f"Error: unsupported loss " + p["loss"])
    
    if p["opt"] == "Adam":
        opt = optim.Adam(model.parameters(), p["lr"])
    
    elif p["opt"] == "SGD":
        opt = optim.SGD(model.parameters(), p["lr"], p["momentum"])
    
    else:
        raise NotImplementedError("Error: unsupported optimizer " + p["opt"])
    

    # Prepare wandb run
    if use_wandb:
        run = wandb.init(config= p, project= "NP-PINN")

    num_epochs = p["num_epochs"]

    # Main training loop
    print("Begin training at " + str(datetime.datetime.now()))
    
    model.train()

    for epoch in tqdm(range(num_epochs)):

        for batch in train_loader:

            coords, data = batch

            train_out = model(coords)

            batch_loss, loss_dict = loss_fn(data, train_out)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            if use_wandb:
                run.log(loss_dict)
        







