from ML.model import ForwardPINN
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Util.evaluate_function import evaluate_env

def model_concplot(model: ForwardPINN, name, time: float, save_ext):

    tt = np.load("./Data/" + name + "_tt.npy")
    xx = np.load("./Data/" + name + "_xx.npy")
    yy = np.load("./Data/" + name + "_yy.npy")

    time_idx = np.where(tt[:,0,0] >= time)[0][0]

    C_N_anal = np.load("./Data/" + name + "_C_N.npy")
    C_F_anal = np.load("./Data/" + name + "_C_F.npy")
    C_INT_anal = np.load("./Data/" + name + "_C_INT.npy")

    coords = np.stack([tt[time_idx].flatten(), xx[time_idx].flatten(), yy[time_idx].flatten()], axis = -1)
    coords = t.from_numpy(coords).float()
    preds = model.forward_unscaled(coords) 

    C_N_preds = preds[:,0]
    C_F_preds = preds[:,1]
    C_INT_preds = preds[:,2]

    C_N_preds_arr = C_N_preds.cpu().detach().numpy().reshape(tt[time_idx].shape)
    C_F_preds_arr = C_F_preds.cpu().detach().numpy().reshape(tt[time_idx].shape)
    C_INT_preds_arr = C_INT_preds.cpu().detach().numpy().reshape(tt[time_idx].shape)

    
    cmax = max(C_N_preds_arr.max(), C_F_preds_arr.max(), C_INT_preds_arr.max())

    tickstep = 1 / model.env.geometry.ds
    x_ticks = np.arange(0, C_N_preds_arr.shape[0], tickstep)
    y_ticks = np.arange(0, C_N_preds_arr.shape[1], tickstep)
    x_tick_labels = range(len(x_ticks))
    y_tick_labels = range(len(y_ticks))

    fig, ax = plt.subplots(2, 3, figsize = (9, 5))

    ax[0,0].imshow(C_N_preds_arr, vmin= 0, vmax = cmax)
    ax[0,0].set_title(r"$C_N$")
    ax[0,0].set_xticks(x_ticks, x_tick_labels)
    ax[0,0].set_yticks(y_ticks, y_tick_labels)

    ax[1,0].imshow(C_N_anal[time_idx], vmin= 0, vmax = cmax)
    ax[1,0].set_xticks(x_ticks, x_tick_labels)
    ax[1,0].set_yticks(y_ticks, y_tick_labels)

    ax[0,1].imshow(C_F_preds_arr, vmin= 0, vmax = cmax)
    ax[0,1].set_title(r"$C_F$")
    ax[0,1].set_xticks(x_ticks, x_tick_labels)
    ax[0,1].set_yticks(y_ticks, y_tick_labels)

    ax[1,1].imshow(C_F_anal[time_idx], vmin= 0, vmax = cmax)
    ax[1,1].set_xticks(x_ticks, x_tick_labels)
    ax[1,1].set_yticks(y_ticks, y_tick_labels)


    img = ax[0,2].imshow(C_INT_preds_arr, vmin= 0, vmax = cmax)
    ax[0,2].set_title(r"$C_{INT}$")
    ax[0,2].set_xticks(x_ticks, x_tick_labels)
    ax[0,2].set_yticks(y_ticks, y_tick_labels)

    ax[1,2].imshow(C_INT_anal[time_idx], vmin= 0, vmax = cmax)
    ax[1,2].set_xticks(x_ticks, x_tick_labels)
    ax[1,2].set_yticks(y_ticks, y_tick_labels)

    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label("Concentration")
    fig.suptitle(f"Numerical and Model Predicted Concentrations at t= {time}")
    

    fig.savefig("./Plots/" + save_ext + "_modelpreds.png")

def model_p_plot(model: ForwardPINN, P_i, save_ext):
    env = model.env
    P_arr = evaluate_env(P_i, env.geometry)[0]
    p_max = P_arr.max()
    xvals = t.linspace(0, env.geometry.width, env.geometry.shape_x)
    yvals = t.linspace(0, env.geometry.height, env.geometry.shape_y)

    coords = t.meshgrid(xvals, yvals, indexing= "ij")
    coords = t.stack(coords, dim= -1).reshape(-1,2)

    preds = model(coords).cpu().detach().numpy()
    preds = preds.reshape(env.geometry.shape_x, env.geometry.shape_y)

    tickstep = 1 / model.env.geometry.ds
    x_ticks = np.arange(0, preds.shape[0], tickstep)
    y_ticks = np.arange(0, preds.shape[1], tickstep)
    x_tick_labels = range(len(x_ticks))
    y_tick_labels = range(len(y_ticks))

    fig, ax = plt.subplots(2, figsize= (4.8, 6.4))

    ax[0].imshow(preds, vmin= 0, vmax= p_max)
    ax[0].set_title("Predicted (PyTorch)")
    ax[0].set_xticks(x_ticks, x_tick_labels)
    ax[0].set_yticks(y_ticks, y_tick_labels)

    img = ax[1].imshow(P_arr, vmin= 0, vmax= p_max)
    ax[1].set_title("Numerical (Fenicsx)")
    ax[1].set_xticks(x_ticks, x_tick_labels)
    ax[1].set_yticks(y_ticks, y_tick_labels)

    fig.colorbar(img, ax=ax, orientation='vertical')
    fig.suptitle("Pressure Profile Comparison")

    fig.savefig("./Plots/" + save_ext + "_pressure_model.png")

def model_p_lineplot(model: ForwardPINN, P_i, save_ext):
    env = model.env
    P_arr = evaluate_env(P_i, env.geometry)[0]
    p_max = P_arr.max()
    xvals = t.linspace(0, env.geometry.width, env.geometry.shape_x)
    midpoint = t.tensor(env.geometry.height / 2)
    midpoint_t = t.tile(midpoint, (len(xvals),))
    
    coords = t.stack([xvals, midpoint_t], dim= 1)

    preds = model(coords).cpu().detach().numpy()

    anal_midpoint = int(midpoint.item() / model.env.geometry.ds)

    anal_vals = P_arr[:, anal_midpoint]

    plt.plot(xvals, anal_vals, linewidth= 0.5, label = "Numerical (Fenicsx)")
    plt.plot(xvals, preds, linewidth= 0.5, label= "Predicted (PyTorch)")
    plt.title("Pressure profile comparison")
    plt.xlabel("x (cm)")
    plt.ylabel(r"$P_i$ (mmHg)")
    plt.legend()
    plt.savefig("./Plots/" + save_ext + "_pressure_model_lineplot.png")
    
def model_conc_anim(model: ForwardPINN, name, save_ext, fps= 30):
    # Load data
    tt = np.load(f"./Data/{name}_tt.npy")
    xx = np.load(f"./Data/{name}_xx.npy")
    yy = np.load(f"./Data/{name}_yy.npy")

    C_N_anal = np.load(f"./Data/{name}_C_N.npy")
    C_F_anal = np.load(f"./Data/{name}_C_F.npy")
    C_INT_anal = np.load(f"./Data/{name}_C_INT.npy")

    # Tick parameters
    tickstep = 1 / model.env.geometry.ds
    x_ticks = np.arange(0, tt.shape[1], tickstep)
    y_ticks = np.arange(0, tt.shape[2], tickstep)
    x_tick_labels = range(len(x_ticks))
    y_tick_labels = range(len(y_ticks))

    # Color scale across all time steps
    cmax = max(C_N_anal.max(), C_F_anal.max(), C_INT_anal.max())

    fig, ax = plt.subplots(2, 3, figsize=(9, 5))

    # --- First Row: Predicted ---
    im_pred_N = ax[0, 0].imshow(np.zeros_like(tt[0]), vmin=0, vmax=cmax, cmap= "Blues")
    ax[0, 0].set_title(r"$C_N$")
    ax[0, 0].set_ylabel("Predicted (PyTorch)")
    ax[0, 0].set_xticks(x_ticks, x_tick_labels)
    ax[0, 0].set_yticks(y_ticks, y_tick_labels)

    im_pred_F = ax[0, 1].imshow(np.zeros_like(tt[0]), vmin=0, vmax=cmax, cmap= "Blues")
    ax[0, 1].set_title(r"$C_F$")
    ax[0, 1].set_xticks(x_ticks, x_tick_labels)
    ax[0, 1].set_yticks(y_ticks, y_tick_labels)

    im_pred_INT = ax[0, 2].imshow(np.zeros_like(tt[0]), vmin=0, vmax=cmax, cmap= "Blues")
    ax[0, 2].set_title(r"$C_{INT}$")
    ax[0, 2].set_xticks(x_ticks, x_tick_labels)
    ax[0, 2].set_yticks(y_ticks, y_tick_labels)

    # --- Second Row: FEM ---
    im_anal_N = ax[1, 0].imshow(C_N_anal[0], vmin=0, vmax=cmax, cmap= "Blues")
    ax[1, 0].set_ylabel("FEM (FEniCSx)")
    ax[1, 0].set_xticks(x_ticks, x_tick_labels)
    ax[1, 0].set_yticks(y_ticks, y_tick_labels)

    im_anal_F = ax[1, 1].imshow(C_F_anal[0], vmin=0, vmax=cmax, cmap= "Blues")
    ax[1, 1].set_xticks(x_ticks, x_tick_labels)
    ax[1, 1].set_yticks(y_ticks, y_tick_labels)

    im_anal_INT = ax[1, 2].imshow(C_INT_anal[0], vmin=0, vmax=cmax, cmap= "Blues")
    ax[1, 2].set_xticks(x_ticks, x_tick_labels)
    ax[1, 2].set_yticks(y_ticks, y_tick_labels)

    # Add colorbar to last predicted column
    cbar = fig.colorbar(im_pred_INT, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label("Concentration")

    # Animation update function
    def update(frame_idx):
        coords = np.stack([tt[frame_idx].flatten(), xx[frame_idx].flatten(), yy[frame_idx].flatten()], axis=-1)
        coords = t.from_numpy(coords).float()
        preds = model.forward_unscaled(coords).cpu()

        C_N_pred = preds[:, 0].detach().numpy().reshape(tt[frame_idx].shape)
        C_F_pred = preds[:, 1].detach().numpy().reshape(tt[frame_idx].shape)
        C_INT_pred = preds[:, 2].detach().numpy().reshape(tt[frame_idx].shape)

        # Update imshow objects
        im_pred_N.set_data(C_N_pred)
        im_anal_N.set_data(C_N_anal[frame_idx])

        im_pred_F.set_data(C_F_pred)
        im_anal_F.set_data(C_F_anal[frame_idx])

        im_pred_INT.set_data(C_INT_pred)
        im_anal_INT.set_data(C_INT_anal[frame_idx])

        fig.suptitle(f"Model and Numerical Values at t = {tt[frame_idx, 0, 0]:.2f}")
        return [im_pred_N, im_anal_N, im_pred_F, im_anal_F, im_pred_INT, im_anal_INT]

    # Build and save animation
    anim = FuncAnimation(fig, update, frames=range(len(tt)), interval=1000 / fps)

    anim.save(f"./Animations/{save_ext}_model_concs.mp4", fps=fps, dpi=150)

    plt.close(fig)


    



        
        
        



        
        
        
