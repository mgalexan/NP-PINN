from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.flags import SphericalFlag, EdgeFlag2D
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc_valid import calculate_concentrations
from Util.interpreter import Interpreter
from Util.evaluate_function import evaluate_env
import numpy as np
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)


from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()


    
name = "four_conc"

test_geo = GeometrySpace(5, 5, 0, 0.025, 0.1, 3600)

test = ParamSpace(test_geo)


test.open_params("./Config/sim_params_valid.json")


test.add_flag(SphericalFlag([2.5, 2.5], 1.0))
test.add_flag(EdgeFlag2D(0.1), "edge")
test.add_flag(SphericalFlag([2.5, 2.5], 0.6), "necrotic")

if rank == 0:
    print("Refining Mesh...", flush=True)
test.refine_near_tumor(n_iter= 1)
if rank == 0:
    print("Done!", flush=True)

test.broadcast_serial_mesh()

#test_geo.visualize_mesh("test")

if rank == 0:
    print("Computing Pressure...", flush=True)
P_i = calculate_pressure(test, "neumann")
if rank == 0:
    print("Done!", flush=True)
#print(test.flag_locs["edge"])

if rank == 0:
    print("Computing Concentrations...", flush=True)
C = calculate_concentrations(test, P_i, "dirichlet", verbose= True)
if rank == 0:
    print("Done!", flush=True)

    print("Interpreting Results...", flush=True)

labels = ["C_N", "C_F", "C_B", "C_INT"]
labels_tex = [r"$C_N$", r"$C_F$", r"$C_B$", r"$C_{INT}$"]

interp = Interpreter(test, C, P_i, sample_rate= 100, labels=labels, labels_tex=labels_tex)

comm.barrier()

if rank == 0:
    #interp.crop([3, 3], 2)

    #interp.save_matrix(name)
    #interp.save_tensor(name)
    print("Done!", flush=True)


    print("Making Plots...", flush=True)

    interp.pressure_plot(name)
    interp.image_animation(name)
    interp.time_center_plots(name, True)


    print("Done!", flush=True)
