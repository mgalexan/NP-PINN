from Environment.geometry import GeometrySpace
from Environment.env_class import ParamSpace
from Environment.tumors import SphericalTumor
from Physics.calculate_pressure import calculate_pressure
from Physics.calculate_conc import calculate_concentrations
from Util.interpreter import Interpreter

name = "test"

test_geo = GeometrySpace(7, 7, 0, 0.125)

test = ParamSpace(test_geo)

test.open_params("./Config/sim_params.json")

test.add_tumor(SphericalTumor([3.5, 3.5], 1.5))

print("Computing Pressure...", flush=True)
P_i = calculate_pressure(test, "neumann")
print("Done!", flush=True)

dt, T = 0.05, 9000
print("Computing Concentrations...", flush=True)
C = calculate_concentrations(test, dt, T, P_i, "neumann")
print("Done!", flush=True)

print("Interpreting Results...", flush=True)
interp = Interpreter(test, C, P_i, dt, T)
print("Done!", flush=True)

interp.crop([3.5, 3.5], 2)
print("Making Plots...", flush=True)

interp.time_center_plots(name)
interp.pressure_plot(name)
interp.image_animation(name, dt)

print("Done!", flush=True)