from dolfinx import fem
import numpy as np
from Environment.env_class import ParamSpace
from Physics.equations import p_anal
from Environment.geometry import GeometrySpace
from Util.evaluate_function import evaluate_env

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.animation import FuncAnimation
plt.style.use("ggplot")

class Interpreter():

    def __init__(self, env: ParamSpace, C: tuple, P_i: fem.Function, dt: float, T: float, sample_rate = 100):

        self.env = env
        self.geometry = env.geometry
        self.dim = self.geometry.dim
        self.sample_rate = sample_rate
        
        self.load_P(P_i)
        self.load_C(C, sample_rate)

        self.midpoint = self.geometry.shape_x // 2

        self.tvals = np.linspace(0, T, len(self.C_N_vals))
        self.xvals = np.linspace(0, self.geometry.width, self.geometry.shape_x)
        self.yvals = np.linspace(0, self.geometry.height, self.geometry.shape_y)
        
        self.labels = ["C_N", "C_F", "C_INT"]
        self.labels_tex = [r"$C_N$", r"$C_F$", r"$C_{INT}$"]

    def crop(self, center: list, width: float):
        
        idx_center = [int(center[i] / self.geometry.ds) for i in range(len(center))]
        width_idx = int(width / self.geometry.ds)

        if self.dim == 1:
            self.P_i_val = self.P_i_val[idx_center[0] - width_idx[0]:idx_center[0] + width_idx[0] + 1]
            for i in range(3):
                self.C_vals[i] = [C[idx_center[0] - width_idx[0]:idx_center[0] + width_idx[0] + 1] for C in self.C_vals[i]]
            self.C_N_vals = self.C_vals[0]
            self.C_F_vals = self.C_vals[1]
            self.C_INT_vals = self.C_vals[2]
            self.xvals = np.linspace(0, 2 * width, self.P_i_val.shape[0])
        
        if self.dim == 2:
            self.P_i_val = self.P_i_val[idx_center[0] - width_idx:idx_center[0] + width_idx + 1, idx_center[1] - width_idx:idx_center[1] + width_idx + 1]
            for i in range(3):
                self.C_vals[i] = [C[idx_center[0] - width_idx:idx_center[0] + width_idx + 1, idx_center[1] - width_idx:idx_center[1] + width_idx + 1] for C in self.C_vals[i]]
            self.C_N_vals = self.C_vals[0]
            self.C_F_vals = self.C_vals[1]
            self.C_INT_vals = self.C_vals[2]
            self.xvals = np.linspace(0, 2 * width, self.P_i_val.shape[0])
            self.yvals = np.linspace(0, 2 * width, self.P_i_val.shape[1])

        self.midpoint = self.P_i_val.shape[0] // 2

    def load_P(self, P_i: fem.Function):
        self.P_i = P_i
        self.P_i_val = evaluate_env(P_i, self.geometry)[0]

    def load_C(self, C: tuple, sample_rate = 100):
        self.C_N = C[0]
        self.C_F = C[1]
        self.C_INT = C[2]
        self.C = [self.C_N, self.C_F, self.C_INT]

        self.C_N_vals = [evaluate_env(C, self.geometry)[0] for C in self.C_N]
        self.C_F_vals = [evaluate_env(C, self.geometry)[0] for C in self.C_F]
        self.C_INT_vals = [evaluate_env(C, self.geometry)[0] for C in self.C_INT]
        self.C_vals = [self.C_N_vals, self.C_F_vals, self.C_INT_vals]
    
    def pressure_plot(self, save_ext: str):
        
        if self.dim == 1:
            plt.plot(self.xvals, self.P_i_val)
            plt.xlabel("x (cm)")
            plt.ylabel(r"$P_i$ (mmHg)")
        
        elif self.dim == 2:
            tickstep = 1 / self.geometry.ds
            x_ticks = np.arange(0, self.P_i_val.shape[0], tickstep)
            y_ticks = np.arange(0, self.P_i_val.shape[1], tickstep)
            x_tick_labels = range(len(x_ticks))
            y_tick_labels = range(len(y_ticks))

            fig = plt.imshow(self.P_i_val)
            plt.colorbar(fig, label= r"$P_i$ (mmHg)")
            plt.xticks(x_ticks, x_tick_labels)
            plt.yticks(y_ticks, y_tick_labels)
            plt.xlabel("x (cm)")
            plt.ylabel("y (cm)")
            
        plt.title("Spatial Pressure in the Tumor Microvenvironment")
        plt.savefig("./Plots/" + save_ext + "_pressure.png")
        plt.clf()

    def time_center_plots(self, save_ext: str):

        for i in range(3):
            if self.dim == 1:
                sim_vals = np.array(self.C_vals[i])[:, self.midpoint]
            elif self.dim == 2:
                sim_vals = np.array(self.C_vals[i])[:, self.midpoint, self.midpoint]

            plt.plot(self.tvals, sim_vals, linewidth= 0.5)
            plt.title(r"Evolution of " + self.labels_tex[i] + r"at tumor center by time")
            plt.xlabel("time (s)")
            plt.ylabel(self.labels_tex[i])
            plt.savefig("./Plots/" + save_ext + "_conc_time_" + self.labels[i] + ".png")
            plt.clf()

    def pressure_analytic_comparison(self, R, save_ext: str):
        
        r_vals = self.xvals[0: self.midpoint + 1]
        
        if self.dim == 1:
            num_vals = self.P_i_val[self.midpoint:]
        elif self.dim == 2:
            num_vals = self.P_i_val[self.midpoint:, self.midpoint]
        
        anal_vals = p_anal(r_vals / R, self.env.params, R)

        plt.plot(r_vals, num_vals, label= "Numerical (FEniCSx)", linewidth=0.5)
        plt.plot(r_vals, anal_vals, label= "Analytic", linewidth= 0.5)
        plt.legend()
        plt.xlabel("r (cm)")
        plt.ylabel("P (mmHg)")
        plt.title(r"Analytic and Numeric Pressure Comparison")
        plt.savefig("./Plots/" + save_ext + "_P_comparison.png")
        plt.clf()

    def line_animation(self, save_ext: str):
        pass

    def image_animation(self, save_ext: str, dt, fps = 30, rate= 1):
        
        tickstep = 1 / self.geometry.ds
        x_ticks = np.arange(0, self.P_i_val.shape[0], tickstep)
        y_ticks = np.arange(0, self.P_i_val.shape[1], tickstep)
        x_tick_labels = range(len(x_ticks))
        y_tick_labels = range(len(y_ticks))


        for i in range(3):
            max_c = np.array(self.C_vals[i]).max()

            def plot_frame(n):
                plt.clf()
                fig  = plt.imshow(self.C_vals[i][n], vmin= 0, vmax= max_c)
                plt.xticks(x_ticks, x_tick_labels)
                plt.yticks(y_ticks, y_tick_labels)
                plt.title(f"Concentration at time t= {n * dt * self.sample_rate}")
                plt.colorbar(fig, label= self.labels_tex[i])
                plt.xlabel("x (cm)")
                plt.ylabel("y (cm)")
   
            fig, ax = plt.subplots()

            # Wrapper for animation: clears and re-plots each frame
            def update(n):
                ax.clear()
                return plot_frame(n)

            # Create animation: frames = number of timesteps
            ani = FuncAnimation(fig, update, frames=range(0, len(self.C_N_vals), 1), blit=False)

            ani.save("./Animations/"+ save_ext +"_animation_" +  self.labels[i] + ".mp4", fps=30, dpi=150, extra_args=['-vcodec', 'libx264'])
        
        plt.clf()