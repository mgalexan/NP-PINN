from ML.model import BackwardPINN
import torch as t
import numpy as np
import matplotlib.pyplot as plt


def model_implot(model: BackwardPINN, name, time: float, save_ext):

    tt = np.load("./Data/" + name + "_tt.npy")
    xx = np.load("./Data/" + name + "_xx.npy")
    yy = np.load("./Data/" + name + "_yy.npy")

    coords = np.stack([tt.flatten(), xx.flatten(), yy.flatten()], axis = -1)
    coords = t.from_numpy(coords).float()
    print(coords)
    preds = model(coords)

    C_N_preds = preds[:,0]
    C_F_preds = preds[:,1]
    C_INT_preds = preds[:,2]

    C_N_preds_arr = C_N_preds.detach().numpy().reshape(tt.shape)
    C_F_preds_arr = C_F_preds.detach().numpy().reshape(tt.shape)
    C_INT_preds_arr = C_INT_preds.detach().numpy().reshape(tt.shape)

    time_idx = np.where(tt[:,0,0] >= time)[0][0]
    cmax = max(C_N_preds_arr.max(), C_F_preds_arr.max(), C_INT_preds_arr.max())

    tickstep = 1 / model.env.geometry.ds
    x_ticks = np.arange(0, C_N_preds_arr.shape[1], tickstep)
    y_ticks = np.arange(0, C_N_preds_arr.shape[2], tickstep)
    x_tick_labels = range(len(x_ticks))
    y_tick_labels = range(len(y_ticks))

    fig, ax = plt.subplots(1, 3, figsize = (6.4, 2.8))

    ax[0].imshow(C_N_preds_arr[time_idx], vmin= 0, vmax = cmax)
    ax[0].set_title(r"$C_N$")
    ax[0].set_xticks(x_ticks, x_tick_labels)
    ax[0].set_yticks(y_ticks, y_tick_labels)
    ax[1].imshow(C_F_preds_arr[time_idx], vmin= 0, vmax = cmax)
    ax[1].set_title(r"$C_F$")
    ax[1].set_xticks(x_ticks, x_tick_labels)
    ax[1].set_yticks(y_ticks, y_tick_labels)
    img = ax[2].imshow(C_INT_preds_arr[time_idx], vmin= 0, vmax = cmax)
    ax[2].set_title(r"$C_{INT}$")
    ax[2].set_xticks(x_ticks, x_tick_labels)
    ax[2].set_yticks(y_ticks, y_tick_labels)
    cbar = fig.colorbar(img, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label("Concentration")
    fig.suptitle(f"Model Predicted Concentrations at t= {time}")
    

    fig.savefig("./Plots/" + save_ext + "_modelpreds.png")


    
