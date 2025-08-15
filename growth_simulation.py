from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.flags import SphericalFlag, EdgeFlag2D, SphericalTaperingFlag
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_growth import calculate_growth
from Util.interpreter import Interpreter
from Util.evaluate_function import evaluate_env
import numpy as np
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)


from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()


    
name = "test_growth"

test_geo = GeometrySpace(4, 4, 0, 0.02, 0.01, 1)
test_geo.get_mesh()

test = ParamSpace(test_geo)


test.open_params("./Config/growth_params.json")


test.add_flag(SphericalFlag([2, 2], 0.1))
test.compile_flags()


test.broadcast_serial_mesh()



if rank == 0:
    print("Computing Growth...", flush=True)
N = calculate_growth(test, "neumann", sample_rate=1, verbose= True)
if rank == 0:
    print("Done!", flush=True)

    print("Interpreting Results...", flush=True)

labels = ["N"]
labels_tex = [r"$N$"]

interp = Interpreter(test, (N,), sample_rate= 100, labels=labels, labels_tex=labels_tex)

comm.barrier()

if rank == 0:
    
    print("Done!", flush=True)


    print("Making Plots...", flush=True)

    
    interp.image_animation(name)


    print("Done!", flush=True)
