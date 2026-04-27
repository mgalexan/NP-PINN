from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.flags import SphericalFlag, EdgeFlag2D, SphericalTaperingFlag
from Physics.calculate_pressure_radial import calculate_pressure
from Physics.calculate_conc_radial import calculate_concentrations
from Util.interpreter import Interpreter
from Util.evaluate_function import evaluate_env
import numpy as np
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)


from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()


    
name = "radial"

test_geo = GeometrySpace(2.0, 0, 0, 0.002, 0.005, 72000)

test = ParamSpace(test_geo)


test.open_params("./Config/sim_params.json")


test.add_flag(SphericalFlag([0.0], 1.0), "tumor")
test.add_flag(SphericalFlag([2.0], 1.0), "edge")

if rank == 0:
    print("Refining Mesh...", flush=True)
test.refine_near_tumor(n_iter= 0)
if rank == 0:
    print("Done!", flush=True)

test.broadcast_serial_mesh()


if rank == 0:
    print("Computing Pressure...", flush=True)
P_i = calculate_pressure(test, "neumann")
if rank == 0:
    print("Done!", flush=True)


if rank == 0:
    print("Computing Concentrations...", flush=True)
C = calculate_concentrations(test, P_i, "neumann", sample_rate=10000, verbose= False, spherical=True)
if rank == 0:
    print("Done!", flush=True)

    print("Interpreting Results...", flush=True)

labels = ["C_N", "C_F", "C_INT"]
labels_tex = [r"$C_N$", r"$C_F$", r"$C_{INT}$"]

interp = Interpreter(test, C, P_i, sample_rate= 10000, labels=labels, labels_tex=labels_tex)

comm.barrier()

if rank == 0:
    #interp.crop([0], 1.0)

    interp.save_matrix(name)
    interp.save_tensor(name)
    print("Done!", flush=True)


    print("Making Plots...", flush=True)

    #interp.pressure_analytic_comparison(1.5, name)
    interp.pressure_plot(name)
    #interp.time_center_plots(name, False)
    interp.line_animation(name)


    print("Done!", flush=True)
