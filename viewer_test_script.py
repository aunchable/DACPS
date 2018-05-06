import numpy as np
import tkinter as tk
from math import *

from simulator.Visualizer import Visualizer
from simulator.ColloidalSystem import ColloidalSystem

worldsize = [800, 800]

type_infos = [
    # id, radius, propensity
    ["jeb", 10, 420],
    ["shiet", 20, 420],
    ["goteem", 30, 420]
]

type_counts = [1, 1, 1]

lj_corr_matrix = [
    [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())], 
    [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())], 
    [(np.random.random(), np.random.random()), (np.random.random(), np.random.random()), (np.random.random(), np.random.random())]
]

#TODO: not do this
target_assembly = None 

cs = ColloidalSystem(worldsize,
                     type_infos,
                     type_counts,
                     lj_corr_matrix,
                     target_assembly)

# initial positions
cs.set_state(np.array([
    [100, 600, 0, 0, 0, 0],
    [400, 100, 0, 0, 0, 0],
    [600, 400, 0, 0, 0, 0],
]))

# let some time pass
for i in range(10):
    cs.step(1, [1, 1, 1])

# wut it look like tho
viz = Visualizer(cs)

viz.update()