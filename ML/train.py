from ML.data_processing import ConcData
from ML.model import ForwardPINN, MLParams, SplitModel
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
    # Prepare wandb run
    if use_wandb:
        run = wandb.init(config= p, project= "NP-PINN")

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
            if use_wandb:
                run.log(loss_dict)
            return total_loss

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
            if use_wandb:
                    run.log(loss_dict)

            return total_loss

    elif p["loss"] == "Conc_Loss_Forward":

        data_loss_fn = MSELoss()

        p_sim = model.env.torch_funcs
        
        coords = model.coloc

        P_i = p_sim["P_i"](coords)
        v_i = -p_sim["v_i"](coords) * p_sim["kappa"](coords)
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

    
        im = plt.scatter(coords[:, 1].cpu().detach().numpy(), coords[:, 2].cpu().detach().numpy(), c= phi_B.cpu().numpy())
        cbar = plt.colorbar(im,)
        cbar.set_label(r"$phi_B$")
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        plt.title(r"Colocation Values of $\phi_B$")
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
            if use_wandb:
                    run.log(loss_dict)

            return total_loss

    elif p["loss"] == "Conc_Loss_Backward":

        data_loss_fn = MSELoss()

        p_sim = model.env.torch_funcs
        
        coords = model.coloc

        P_i = p_sim["P_i"](coords)
        v_i = -p_sim["v_i"](coords) * p_sim["kappa"](coords).detach()
        div_v_i = divergence(v_i, coords).detach()
        P_i = P_i.detach()
        v_i = v_i.detach()
        tumor = p_sim["tumor"](coords).detach()
        SV = p_sim["S/V"](coords).detach()
        D_F = p_sim["D_F"](coords).detach()
        K_INT = p_sim["K_INT"](coords).detach()
        K_degF = p_sim["K_deg-F"](coords).detach()
        K_degINT = p_sim["K_deg-INT"](coords).detach()
        phi_B = compute_phi_B(P_i, coords, p_sim).detach()
        phi_L = compute_phi_L(P_i, coords, p_sim).detach()
        p_nano = MLParams("./Config/nano_physics.json")
        
        def loss_fn(data, train_out):


            data_loss = data_loss_fn(data, train_out)
            if p["phys_weight"] > 0: 
                vals = model.forward_unscaled(coords)
                C_N = vals[:,0].unsqueeze(-1)
                C_F = vals[:,1].unsqueeze(-1)
                C_INT = vals[:,2].unsqueeze(-1)

                #D_N = p_sim["D_N"](coords).detach()
                K_rel = p_sim["K_rel"](coords).detach()
                P = p_sim["P"](coords).detach()
                tau = p_sim["tau"](coords).detach()
                sigma_f = p_sim["sigma_f"](coords).detach()
                alpha = 20#model.alpha * 10
                D_tumor = model.d * 10e-9
                D_normal = 4.52e-8
                #P_normal = 2.02e-8

                


                #sigma_f, P_tumor, K_rel, D_tumor = nano_physics(d, alpha, p_nano, device)

                #print(sigma_f, P_tumor, K_rel, D_tumor)
                
                #tau_dimless = model.tau
                #T0 = 10000
                #tau = T0 * t.exp(tau_dimless)
                D_N = p_sim["tumor"](coords) * (D_normal - D_tumor) + D_normal
                #P = p_sim["tumor"](coords) * (P_normal - P_tumor) + P_normal
                
                Pe_ratio = compute_Pe_ratio(SV, P, sigma_f, phi_B)

                Phi_C = compute_Phi_C(P, sigma_f, tau, Pe_ratio, phi_B, SV, tumor, coords)
                Phi_N = compute_Phi_N(P, Pe_ratio, phi_L, SV, tumor, coords, p_sim)

                
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
                    "C_INT_loss" : C_INT_loss.item(),
                    "d" : D_tumor,
                    #"alpha" : alpha.item()
                    #"K_rel" : model.k_rel.item(),
                    #"tau" : tau.item()
                    #"sigma_f" : sigma_f.item()
                }
            
            else: 
                total_loss = data_loss
                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                }
            if use_wandb:
                    run.log(loss_dict)

            return total_loss
    
    elif p["loss"] == "Conc_Loss_CN":

        data_loss_fn = MSELoss()

        p_sim = model.env.torch_funcs
        
        coords = model.coloc

        P_i = p_sim["P_i"](coords)
        v_i = -p_sim["v_i"](coords) * p_sim["kappa"](coords).detach()
        div_v_i = divergence(v_i, coords).detach()
        P_i = P_i.detach()
        v_i = v_i.detach()
        tumor = p_sim["tumor"](coords).detach()
        SV = p_sim["S/V"](coords).detach()
        phi_B = compute_phi_B(P_i, coords, p_sim).detach()
        phi_L = compute_phi_L(P_i, coords, p_sim).detach()
        p_nano = MLParams("./Config/nano_physics.json")
        
        def loss_fn(data, train_out):

            data_loss = data_loss_fn(data[:,0].unsqueeze(0), train_out[:,0].unsqueeze(0))
            if p["phys_weight"] > 0: 
                vals = model.forward_unscaled(coords)
                C_N = vals[:,0].unsqueeze(-1)
                C_F = vals[:,1].unsqueeze(-1)
                C_INT = vals[:,2].unsqueeze(-1)

                #D_N = p_sim["D_N"](coords).detach()
                K_rel = p_sim["K_rel"](coords).detach()
                P = p_sim["P"](coords).detach()
                tau = p_sim["tau"](coords).detach()
                sigma_f = p_sim["sigma_f"](coords).detach()
                D_tumor = model.d * 10e-9
                # D_normal = 0.0
                #P_normal = 2.02e-8

                #sigma_f, P_tumor, K_rel, D_tumor = nano_physics(d, alpha, p_nano, device)

                #print(sigma_f, P_tumor, K_rel, D_tumor)
                
                #tau_dimless = model.tau
                #T0 = 10000
                #tau = T0 * t.exp(tau_dimless)
                D_N = p_sim["tumor"](coords) * D_tumor
                #P = p_sim["tumor"](coords) * (P_normal - P_tumor) + P_normal
                
                Pe_ratio = compute_Pe_ratio(SV, P, sigma_f, phi_B)

                Phi_C = compute_Phi_C(P, sigma_f, tau, Pe_ratio, phi_B, SV, tumor, coords)
                Phi_N = compute_Phi_N(P, Pe_ratio, phi_L, SV, tumor, coords, p_sim)

                
                C_N_loss = p["phys_weight"] * C_N_Loss(coords, C_N, D_N, v_i, div_v_i, K_rel, Phi_C, Phi_N)

                total_loss = data_loss + C_N_loss

                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                    "C_N_loss" : C_N_loss.item(),
                    "d" : D_tumor,
                    #"alpha" : alpha.item()
                    #"K_rel" : model.k_rel.item(),
                    #"tau" : tau.item()
                    #"sigma_f" : sigma_f.item()
                }
            
            else: 
                total_loss = data_loss
                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                }
            if use_wandb:
                    run.log(loss_dict)

            return total_loss

    elif p["loss"] == "Growth_Loss_Forward":

        data_loss_fn = MSELoss()

        p_sim = model.env.torch_funcs
        
        coords = model.coloc
       
        D = p_sim["D"](coords).detach()
        K = p_sim["K"](coords).detach()
        rho = p_sim["rho"](coords).detach()

        def loss_fn(data, train_out):

            data_loss = data_loss_fn(data, train_out)
            if p["phys_weight"] > 0: 
                N = model.forward_unscaled(coords)
                
                
                N_loss = p["phys_weight"] * N_Loss(coords, N, rho, K, D)
                

                total_loss = data_loss + N_loss

                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                    "N_loss" : N_loss.item(),
                }
            
            else: 
                total_loss = data_loss
                loss_dict = {
                    "total_loss" : total_loss.item(),
                    "log_loss" : np.log10(total_loss.item()),
                    "data_loss" : data_loss.item(),
                }
                if use_wandb:
                    run.log(loss_dict)

            return total_loss

    elif p["loss"] == "Pressure_Loss_Radial":
        data_loss_fn = MSELoss()
        def loss_fn(data, train_out):
            data_loss = data_loss_fn(data, train_out)
            phys_loss =  p["phys_weight"] * pressure_phys_loss_radial(model(model.coloc), model.coloc, model.env.torch_funcs)
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
            if use_wandb:
                    run.log(loss_dict)

            return total_loss
    
    

    else:
        raise NotImplementedError(f"Error: unsupported loss " + p["loss"])
    
    if p["opt"] == "Adam":
        opt = optim.Adam(model.parameters(), p["lr"])
    
    elif p["opt"] == "SGD":
        opt = optim.SGD(model.parameters(), p["lr"], p["momentum"])

    elif p["opt"] == "LBFGS":
        opt = optim.LBFGS(model.parameters(), p["lr"], history_size= 10)
    
    else:
        raise NotImplementedError("Error: unsupported optimizer " + p["opt"])
    
    #scheduler = t.optim.lr_scheduler.ExponentialLR(opt, 0.997)
    scheduler = t.optim.lr_scheduler.ExponentialLR(opt, 1.00)
    

    num_epochs = p["num_epochs"]

    # Main training loop
    print("Begin training at " + str(datetime.datetime.now()))
    
    if p["load_checkpoint"]:
        state = t.load("./Models/checkpoint_model.pt")
        model.load_state_dict(state)
        print("Loaded Checkpoint")
    
    model.train()

    t.autograd.set_detect_anomaly(True)

    best_loss = 1000

    if p["phys_start"] > 0:
        phys_weight = p["phys_weight"]
        p["phys_weight"] = 0

    b_count = 0
    for epoch in tqdm(range(num_epochs), disable= not(verbose)):

        for batch in train_loader:

            locs, data = batch

            data_scaled = data * p["output_scaling"]

            train_out = model(locs)
            
            
            if p["opt"] == "LBFGS":
                def closure():
                    opt.zero_grad()
                    train_out = model(locs)
                    loss = loss_fn(train_out, data_scaled)
                    loss.backward()
                    return loss
                batch_loss = opt.step(closure)

            else:
                batch_loss = loss_fn(data_scaled, train_out)
                opt.zero_grad()
                batch_loss.backward()
                opt.step()

            
            if p["save_best"]:
                if (batch_loss.item() < best_loss) & (epoch > p["phys_start"]):
                    t.save(model.state_dict(), "./Models/checkpoint_model.pt")
                    best_loss = batch_loss.item()
            if p["phys_start"] <= b_count <= 2000 and p["opt"] != "LBFGS":
                scheduler.step()
                b_count += 1
        if not(verbose):
            if epoch % (num_epochs // 100) == 0:
                print(f"Progess {epoch * 100 // num_epochs}%", end= "\r", flush= True)   
        if p["phys_start"] == epoch:
            p["phys_weight"] = phys_weight
        
        

        







