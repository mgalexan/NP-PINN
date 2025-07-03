from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations
from Util.evaluate_function import evaluate_env

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use("ggplot")
np.set_printoptions(threshold=sys.maxsize)


test_geo = GeometrySpace(10, 10, 0, 0.05)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5, 5])

test.add_tumor(SphericalTumor(center1, 4.5))

P_i = calculate_pressure(test, "neumann")

sim_vals, _ = evaluate_env(P_i, test_geo)

xvals = np.linspace(0, 10, 200)

fig = plt.imshow(sim_vals)
plt.colorbar(fig)
plt.savefig("./Plots/test_fig.png")
plt.clf()
'''
p = test.params

R = 5
alpha_t = R * np.sqrt(p["L_P"]["tumor"] * p["S/V"]["tumor"] / p["kappa"]["tumor"])
alpha_h = R * np.sqrt(p["L_P"]["normal"] * p["S/V"]["normal"] / p["kappa"]["normal"])
p_et = p["P_b"] - p["sigma_s"]["tumor"] * (p["pi_b"]["tumor"] - p["pi_i"]["tumor"])
p_eh = p["P_b"] - p["sigma_s"]["normal"] * (p["pi_b"]["normal"] - p["pi_i"]["normal"])
p_e = p_eh/p_et

K = p["kappa"]["tumor"] / p["kappa"]["normal"]

phi = (1 + alpha_h) * np.sinh(alpha_t)
theta = K * (alpha_t * np.sinh(alpha_h) - np.sinh(alpha_h))

def p_anal(r):

    p = np.empty(r.shape)
    

    p[np.where(r <= 1)] = 1 - (1 - p_e) * (alpha_h + 1) * np.sinh(alpha_t * r[np.where(r <= 1)]) / (r[np.where(r <= 1)] * (theta + phi))
    p[np.where(r > 1)] = p_e + (1 - p_e) * theta * np.exp(-alpha_h * (r[np.where(r > 1)] - 1)) / (r[np.where(r > 1)] * (theta + phi))
    return p

#print(alpha_h, alpha_t, p_e, K, p_et)


plt.plot(xvals[:100], sim_vals[100:], label= "Numerical (FEniCSx)", linewidth=0.5)
plt.plot(xvals[:100], p_et * p_anal(xvals[:100] / R), label= "Analytic", linewidth= 0.5)
plt.legend()
plt.xlabel("r (cm)")
plt.ylabel("P (mmHg)")
plt.title(r"Analytic and Numeric Pressure (Nanoparticle $L_p$)")
plt.savefig("./Plots/only_tumor_p.png")
plt.clf()
'''

C_N, C_F, C_INT = calculate_concentrations(test, 0.5, 3000, P_i, "neumann")

C_N = [evaluate_env(C, test_geo)[0] for C in C_N[0::100]]
C_F = [evaluate_env(C, test_geo)[0] for C in C_F[0::100]]
C_INT = [evaluate_env(C, test_geo)[0] for C in C_INT[0::100]]


midpoint = 50
C_N_time = np.array(C_N)[:, midpoint, midpoint]

tvals = np.linspace(0, 3000, 60)

plt.plot(tvals, C_N_time, linewidth= 0.5)
plt.title(r"Evolution of $C_N$ at tumor center by time")
plt.xlabel("time (s)")
plt.ylabel(r"C_N")
plt.savefig("./Plots/conc_time_test.png")
plt.clf()

labels = ["C_N", "C_F", "C_INT"]
labels_tex = [r"$C_N$", r"$C_F$", r"$C_{INT}$"]
C = [C_N, C_F, C_INT]
for i in range(3):
    max_c = np.array(C[i]).max()
    def plot_frame(n):
        vals = C[i][n]
        plt.clf()
        line  = plt.imshow(C[i][n], vmin=0, vmax=max_c)
        plt.xticks(range(0,200,20), range(10))
        plt.yticks(range(0,200,20), range(10))
        plt.xlabel
        plt.title(f"Concentration at time t= {n * 5}")
        plt.colorbar(line, label= labels_tex[i])
        plt.xlabel("x (cm)")
        plt.ylabel("y (cm)")
        #plt.ylim((0, max_c))
        #plt.ylabel(labels[i])
        

    fig, ax = plt.subplots()

    # Wrapper for animation: clears and re-plots each frame
    def update(n):
        ax.clear()
        return plot_frame(n)

    # Create animation: frames = number of timesteps
    ani = FuncAnimation(fig, update, frames=range(0, len(C_N), 1), blit=False)

    ani.save("./Animations/" + labels[i] +"_animation_test.mp4", fps=30, dpi=150, extra_args=['-vcodec', 'libx264'])

