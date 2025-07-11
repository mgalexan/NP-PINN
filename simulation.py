from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations
from Util.interpreter import Interpreter
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()


    
name = "mid_sim"

test_geo = GeometrySpace(6, 6, 0, 0.01, 0.05, 300)

test = ParamSpace(test_geo)


test.open_params("./Config/sim_params.json")

test.add_tumor(SphericalTumor([3, 3], 1.5))

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
if rank == 0:
    print("Computing Concentrations...", flush=True)
C = calculate_concentrations(test, P_i, "neumann", verbose= False)
if rank == 0:
    print("Done!", flush=True)

    print("Interpreting Results...", flush=True)

interp = Interpreter(test, C, P_i)

comm.barrier()

if rank == 0:
    interp.crop([3, 3], 2)

    interp.save_matrix(name)
    interp.save_tensor(name)
    print("Done!", flush=True)


    print("Making Plots...", flush=True)

    interp.pressure_plot(name)
    interp.image_animation(name)
    interp.time_center_plots(name)


    print("Done!", flush=True)
