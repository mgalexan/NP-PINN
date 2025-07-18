from ML.model import ForwardPINN
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from Util.evaluate_function import evaluate_env
plt.style.use("ggplot")

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
    preds = model(coords)

    C_N_preds = preds[:,0]
    C_F_preds = preds[:,1]
    C_INT_preds = preds[:,2]

    C_N_preds_arr = C_N_preds.detach().numpy().reshape(tt[time_idx].shape)
    C_F_preds_arr = C_F_preds.detach().numpy().reshape(tt[time_idx].shape)
    C_INT_preds_arr = C_INT_preds.detach().numpy().reshape(tt[time_idx].shape)

    
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

    preds = model(coords).detach().numpy()
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

    preds = model(coords).detach().numpy()

    anal_midpoint = int(midpoint.item() / model.env.geometry.ds)

    anal_vals = P_arr[:, anal_midpoint]

    plt.plot(xvals, anal_vals, linewidth= 0.5, label = "Numerical (Fenicsx)")
    plt.plot(xvals, preds, linewidth= 0.5, label= "Predicted (PyTorch)")
    plt.title("Pressure profile comparison")
    plt.xlabel("x (cm)")
    plt.ylabel(r"$P_i$ (mmHg)")
    plt.legend()
    plt.savefig("./Plots/" + save_ext + "_pressure_model_lineplot.png")
    

   



    
    
    



    
    
    
