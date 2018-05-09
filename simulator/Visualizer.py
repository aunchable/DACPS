# import modules
import numpy as np
import tkinter as tk
from math import *

root = tk.Tk()
# root.wm_title("Colloidal System Simulation")
# root.mainloop()

class Visualizer:

    def __init__(self, colloidal_system):
        self.system = colloidal_system

        # Set up canvas
        self.canvas = tk.Canvas(root, width=self.system.world_dims[0], height=self.system.world_dims[1],
                           borderwidth=0, highlightthickness=0, bg="white")
        self.canvas.grid()


    def update(self):
        state = self.system.get_state()

        for i in range(self.system.num_particles):
            x, y, theta, xdot, ydot, thetadot = state[i]
            #TODO refactor Colloidal System to make these lookups not stupid
            #TODO replace temp values
            r, type_label = 20, "shiet"+str(i)

            self.plot_particle(x, y, theta, xdot, ydot, r, type_label)

        root.wm_title("Colloidal System Simulation")
        root.mainloop()

    def plot_particle(self, x, y, theta, xdot, ydot, r, label):
        self.canvas.create_oval(x-r, y-r, x+r, y+r)
        self.canvas.create_line(x, y, int(x + r * cos(theta)), int(y + r * sin(theta)))
        vdir = atan(ydot/xdot)
        xedge = int(x + r * cos(vdir))
        yedge = int(y + r * sin(vdir))
        if xdot < 0:
            xedge = int(x - r * cos(vdir))
            yedge = int(y - r * sin(vdir))

        self.canvas.create_line(xedge, yedge, xedge + xdot, yedge + ydot, arrow=tk.LAST)
        self.canvas.create_text((x, y), text=label)
