import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

import warnings
warnings.filterwarnings("ignore")
 
def vonNeumann(n):
    grid = np.zeros((n,n), dtype= int)
    grid[11//2,11//2] = 1

    ax = plt.axes()
    ax.set_axis_off()

    ax.imshow(grid, interpolation='none',cmap='RdPu')

    plt.show()
def ruleset(seq,binum):
    if np.array_equal(seq,[1, 1, 1]):
        return binum[0]
    if np.array_equal(seq,[1, 1, 0]):
        return binum[1]
    if np.array_equal(seq,[1, 0, 1]):
        return binum[2]
    if np.array_equal(seq,[1, 0, 0]):
        return binum[3]
    if np.array_equal(seq,[0, 1, 1]):
        return binum[4]
    if np.array_equal(seq,[0, 1, 0]):
        return binum[5]
    if np.array_equal(seq,[0, 0, 1]):
        return binum[6]
    if np.array_equal(seq,[0, 0, 0]):
        return binum[7]
    
def cellular_automation(rule, n,steps,ini_value="",custom_array=np.array):

    rulenum = np.binary_repr(rule,width=8)

    pattern = np.zeros((steps,n), dtype = int)
    #array = list(range(11))
    if ini_value == "random":
        pattern[0] = np.random.randint(2,size=n)
    elif ini_value == "impulse":
        pattern[0,n//2] = 1
    elif ini_value == "custom":
        pattern[0] = custom_array

    for i in range(steps-1):
        for j in range(n):
            seq = np.roll(pattern[i],-j+1)[0:3]
            pattern[i+1,j] = ruleset(seq,rulenum)

    return pattern

def visualization(pattern):
    ax = plt.axes()
    ax.set_axis_off()

    ax.imshow(pattern, interpolation='none',cmap='coolwarm')

    plt.show()

def animate(i):
    #x = cellular_automation(np.random.randint(256),size,5,ini_value="random")
    ax = plt.axes()

    ax.clear()  # clear the plot
    ax.set_axis_off()

    Y = np.zeros((steps_to_show, size), dtype=np.int8)  # initialize with all zeros
    upper_boundary = (i + 1) * iterations_per_frame  # window upper boundary
    lower_boundary = 0 if upper_boundary <= steps_to_show else upper_boundary - steps_to_show  # window lower bound.
    for t in range(lower_boundary, upper_boundary):  # assign the values
        Y[t - lower_boundary, :] = x[t, :]
    
    img = ax.imshow(Y, interpolation='none',cmap=random.choice(colors))
    return [img]

size = 51  # number of cells in one row
steps = 200  # number of time steps
steps_to_show = 20  # number of steps to show in the animation window
iterations_per_frame = 1  # how many steps to show per frame
frames = int(steps // iterations_per_frame)  # number of frames in the animation
interval=8  # interval in ms between consecutive frames
colors= ["plasma", "inferno","coolwarm", "PRGn", "prism"]
colors = ["coolwarm"]

exceptions = [0,8,32,40,64,72,96,128]

def multiple():
    x = cellular_automation(np.random.randint(256),size,5,ini_value="random")
    for i in range(steps//5):
        rule = np.random.randint(256)
        #print(rule)
        if np.all(x[-1]==0) or np.all(x[-1]==1):
            x = np.append(x,cellular_automation(rule,size,5,ini_value="random"),axis=0)

        else:
            x = np.append(x,cellular_automation(rule,size,5,ini_value="custom", custom_array=x[-1]),axis=0)

    return x

fig = plt.figure(figsize=(8, 8))

def call_anim():
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True)

    plt.show()


x = multiple()
call_anim()