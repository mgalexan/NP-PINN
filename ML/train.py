from ML.data_processing import ConcData
from ML.model import ForwardPINN, MLParams
from Physics.physloss import *

import torch as t
import torch.optim as optim
from torch.nn import MSELoss
import numpy as np
import matplotlib.pyplot as plt
import wandb
import datetime
from tqdm import tqdm

def train_model(model: ForwardPINN, p: MLParams, train_loader: t.utils.data.DataLoader, use_wandb: bool = False, verbose = True):

    

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
                "total_loss" : total_loss,
                "log_loss" : total_log_loss,
            }

            return total_loss, loss_dict

    elif p["loss"] == "Pressure_Loss":
        data_loss_fn = MSELoss()
        def loss_fn(data, train_out):
            data_loss = data_loss_fn(data, train_out)
            phys_loss =  p["phys_weight"] * pressure_phys_loss(model(model.coloc), model.coloc, model.env.torch_funcs)
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

    elif p["loss"] == "Conc_Loss_Forward":

        data_loss_fn = MSELoss()

        p_sim = model.env.torch_funcs
        
        coords = model.coloc

        P_i = p_sim["P_i"](coords)
        v_i = -p_sim["v_i"](coords) * p_sim["kappa"](coords).detach()
        div_v_i = divergence(v_i, coords).detach()
        P_i = P_i.detach()
        v_i = v_i.detach()
        
        D_N = p_sim["D_N"](coords).detach()
        K_rel = p_sim["K_rel"](coords).detach()
        P = p_sim["P"](coords).detach()
        sigma_f = p_sim["sigma_f"](coords).detach()
        SV = p_sim["S/V"](coords).detach()
        tau = p_sim["tau"](coords).detach()
        D_F = p_sim["D_F"](coords).detach()
        alpha = p_sim["alpha"](coords).detach()
        K_INT = p_sim["K_INT"](coords).detach()
        K_degF = p_sim["K_deg-F"](coords).detach()
        K_degINT = p_sim["K_deg-INT"](coords).detach()
        tumor = p_sim["tumor"](coords).detach()

        phi_B = compute_phi_B(P_i, coords, p_sim).detach()
        phi_L = compute_phi_L(P_i, coords, p_sim).detach()
        Pe_ratio = compute_Pe_ratio(SV, P, sigma_f, phi_B).detach()

        Phi_C = compute_Phi_C(P, sigma_f, tau, Pe_ratio, phi_B, SV, tumor, coords).detach()
        Phi_N = compute_Phi_N(P, Pe_ratio, phi_L, SV, tumor, coords, p_sim).detach()
        im = plt.scatter(coords[:, 1].detach().numpy(), coords[:, 2].detach().numpy(), c= np.log(Phi_C.numpy()))
        plt.colorbar(im)
        plt.savefig("./Plots/test_scatter.png")
        def loss_fn(data, train_out):

            data_loss = data_loss_fn(data, train_out)
            if p["phys_weight"] > 0: 
                vals = model.forward_unscaled(coords)
                C_N = vals[:,0].unsqueeze(-1)
                C_F = vals[:,1].unsqueeze(-1)
                C_INT = vals[:,2].unsqueeze(-1)

                
                C_N_loss = p["phys_weight"] * C_N_Loss(coords, C_N, D_N, v_i, div_v_i, K_rel, Phi_C, Phi_N)
                C_F_loss = p["phys_weight"] * C_F_Loss(coords, C_F, C_N, C_INT, D_F, v_i, div_v_i, alpha, K_rel, K_INT, K_degINT, K_degF)
                C_INT_loss = p["phys_weight"] * C_INT_Loss(coords, C_INT, C_F, K_degINT, K_INT)

                total_loss = data_loss + C_N_loss + C_F_loss + C_INT_loss

                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                    "C_N_loss" : C_N_loss.item(),
                    "C_F_loss" : C_F_loss.item(),
                    "C_INT_loss" : C_INT_loss.item()
                }
            
            else: 
                total_loss = data_loss
                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
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
    best_loss = 100

    if p["phys_start"] > 0:
        phys_weight = p["phys_weight"]
        p["phys_weight"] = 0

    for epoch in tqdm(range(num_epochs), disable= not(verbose)):

        for batch in train_loader:

            locs, data = batch

            data_scaled = data * p["output_scaling"]

            train_out = model(locs)

            batch_loss, loss_dict = loss_fn(data_scaled, train_out)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            if use_wandb:
                run.log(loss_dict)
            if p["save_best"]:
                if loss_dict["total_loss"] < best_loss:
                    t.save(model.state_dict(), "./Models/checkpoint_model.pt")
                    best_loss = p["save_best"]
        if not(verbose):
            if epoch % (num_epochs // 100) == 0:
                print(f"Progess {epoch * 100 // num_epochs}%", end= "\r")   
        if p["phys_start"] == epoch:
            p["phys_weight"] = phys_weight

        







