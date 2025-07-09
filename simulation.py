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

if rank == 0:

    name = "1d_test"

    test_geo = GeometrySpace(6, 0, 0, 0.0025, 0.05, 900)

    test = ParamSpace(test_geo)


    test.open_params("./Config/sim_params.json")

    test.add_tumor(SphericalTumor([3], 1.5))

test = comm.bcast(test if rank == 0 else None, root=0)

comm.barrier()

print("Computing Pressure...", flush=True)
P_i = calculate_pressure(test, "neumann")
print("Done!", flush=True)

print("Computing Concentrations...", flush=True)
C = calculate_concentrations(test, P_i, "neumann", verbose= True)
print("Done!", flush=True)

print("Interpreting Results...", flush=True)

interp = Interpreter(test, C, P_i)

comm.barrier()

if rank == 0:
    interp.crop([3], 2)

    #interp.save_matrix(name)
    #interp.save_tensor(name)
    print("Done!", flush=True)


    print("Making Plots...", flush=True)

    interp.time_center_plots(name)
    interp.pressure_plot(name)
    interp.line_animation(name)

    print("Done!", flush=True)
