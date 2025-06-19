from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace, save_env, load_env
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations
from Util.evaluate_function import evaluate

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use("ggplot")
np.set_printoptions(threshold=sys.maxsize)


test_geo = GeometrySpace(10, 10, 10, 0.25)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

center1 = np.array([5,5,5])
#center2 = np.array([4, 2, 4])
#center3 = np.array([6,5, 6.5])


test.add_tumor(SphericalTumor(center1, 2))
#test.add_tumor(SphericalTumor(center2, 2))
#test.add_tumor(SphericalTumor(center3, 1))


P_i = calculate_pressure(test, "neumann")

points = test_geo.coord_matrix.reshape(-1, test_geo.dim)
sim_vals, _ = evaluate(P_i, points)


sim_vals = sim_vals.reshape(test_geo.shape)

xvals = np.linspace(0,5, 21)

p = test.params

R = 2
alpha_t = R * np.sqrt(p["L_P"]["tumor"] * p["S/V"]["tumor"] / p["kappa"]["tumor"])
alpha_h = R * np.sqrt(p["L_P"]["normal"] * p["S/V"]["normal"] / p["kappa"]["normal"])
p_et = p["P_b"] - p["sigma_s"]["tumor"] * (p["pi_b"]["tumor"] - p["pi_i"]["tumor"])
p_eh = p["P_b"] - p["sigma_s"]["normal"] * (p["pi_b"]["normal"] - p["pi_i"]["normal"])
p_e = p_eh/p_et

K = p["kappa"]["tumor"] / p["kappa"]["normal"]

phi = (1 + alpha_t) * np.sinh(alpha_t)
theta = K * (alpha_t * np.sinh(alpha_t) - np.sinh(alpha_h))

def p_anal(r):

    p = np.empty(r.shape)
    

    p[np.where(r <= 1)] = 1 - (1 - p_e) * (alpha_h + 1) * np.sinh(alpha_t * r[np.where(r <= 1)]) / (r[np.where(r <= 1)] * (theta + phi))
    p[np.where(r > 1)] = p_e + (1 - p_e) * theta * np.exp(-alpha_h * (r[np.where(r > 1)] - 1)) / (r[np.where(r > 1)] * (theta + phi))
    return p

print(theta, phi, alpha_h, alpha_t)

plt.plot(xvals, sim_vals[20, 20, 20:])
plt.plot(xvals, p_et * p_anal(xvals / R))
plt.savefig("./Plots/test_fig.png")

'''
C_N, C_F, C_INT = calculate_concentrations(test, 0.05, 100, P_i, "neumann")



C_N = [evaluate_function_at_points(C, test_geo.coord_matrix)[0] for C in C_N]
C_F = [evaluate_function_at_points(C, test_geo.coord_matrix)[0] for C in C_F]
C_INT = [evaluate_function_at_points(C, test_geo.coord_matrix)[0] for C in C_INT]

labels = ["C_N", "C_F", "C_INT"]
C = [C_N, C_F, C_INT]
for i in range(3):
    max_c = np.array(C[i]).max()
    def plot_frame(n):
        vals = C[i][n]
        plt.cla()
        line,  = plt.plot(test_geo.coord_matrix, vals, linewidth = 0.5)
        plt.title(f"Concentration at time t= {n * 0.05}")
        plt.xlabel("x (cm)")
        plt.ylabel(labels[i])
        plt.ylim(0, max_c)

    fig, ax = plt.subplots()

    # Wrapper for animation: clears and re-plots each frame
    def update(n):
        ax.clear()
        return plot_frame(n)

    # Create animation: frames = number of timesteps
    ani = FuncAnimation(fig, update, frames=range(0, len(C_N), 50), blit=False)

    ani.save("./Plots/" + labels[i] +"_animation.mp4", fps=30, dpi=150, extra_args=['-vcodec', 'libx264'])
'''

