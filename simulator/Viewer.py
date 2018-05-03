# CS 159 Project 
# Visualizations
# Written by David Wang on 5/2/2018

# import modules
import numpy as np
import tkinter as tk
from math import *



# Initiate random state to test on

worldsize = [800, 800]
num_particles = 10
min_r = 10
max_r = 40
max_v = 40


particles = np.zeros((num_particles, 8))

#initialize positions
for i in [0, 1]:
    particles[:,i] = np.random.randint(0 + max_r + max_v, worldsize[i] - max_r - max_v, num_particles)
    
#initialize theta
particles[:,2] = 2 * np.pi *np.random.random(num_particles)

#initialize velocity
particles[:,3] = np.random.randint(-max_v, max_v, num_particles)
particles[:,4] = np.random.randint(-max_v, max_v, num_particles)

#initialize thetadot
particles[:,5] = np.random.random(num_particles) * 6 - 1

#initialize radii
particles[:,6] = np.random.randint(min_r, max_r, num_particles)

#initialize type
particles[:,7] = (1000000*np.random.random(num_particles)).astype(int)

state = {'particle_vector': particles,
        'world_size': worldsize}



# Monkey patching our own fucntions into tkinter

def _create_circle(self, x, y, r, **kwargs):
    '''
    New tkinter.canvas method to plot a circle at x, y with radius r
    '''
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle

def _plot_orientation(self, x, y, theta, r, **kwargs):
    '''
    New tkinter.canvas method to plot a line from the center to the edge 
    of a circle at x, y, of radius r, at an angle of theta'''
    return self.create_line(x, y, int(x + r * cos(theta)), int(y + r * sin(theta)), **kwargs)
tk.Canvas.plot_orientation = _plot_orientation

def _plot_velocity(self,x, y, xdot, ydot, r, **kwargs):
    '''
    New tkiner.canvas method to plot the velocity vector from the edge of 
    the circle at x, y with radius r'''
    vdir = atan(ydot/xdot)
    xedge = int(x + r * cos(vdir))
    yedge = int(y + r * sin(vdir))
    if xdot < 0:
        xedge = int(x - r * cos(vdir))
        yedge = int(y - r * sin(vdir))
        
    return self.create_line(xedge, yedge, xedge + xdot, yedge + ydot, arrow=tk.LAST, **kwargs)
tk.Canvas.plot_velocity = _plot_velocity




# Set up canvas
root = tk.Tk()
canvas = tk.Canvas(root, width=state['world_size'][0], height=state['world_size'][1],
                   borderwidth=0, highlightthickness=0, bg="white")
canvas.grid() 


for p in state['particle_vector']:
    x, y, theta, xdot, ydot, thetadot, r, tp = p
    canvas.create_circle(x, y, r)#, fill="blue")
    canvas.plot_orientation(x, y, theta, r)
    canvas.plot_velocity(x, y, xdot, ydot, r)
    canvas.create_text((x, y), text=str(int(tp)))



    
root.wm_title("Circles and Arcs")
root.mainloop()





