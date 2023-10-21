import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy
from random import random

x_start, y_start = -2, -1.5  # an interesting region starts here
width, height = 3, 3  # for 3 units up and right
density_per_unit = 250  # how many pixles per unit

# real and imaginary axis
re = np.linspace(x_start, x_start + width, width * density_per_unit )
im = np.linspace(y_start, y_start + height, height * density_per_unit)

fig = plt.figure(figsize=(10, 10))  # instantiate a figure to draw
ax = plt.axes()  # create an axes object

def mandelbrot(x, y, threshold):
    """Calculates whether the number c = x + i*y belongs to the 
    Mandelbrot set. In order to belong, the sequence z[i + 1] = z[i]**2 + c
    must not diverge after 'threshold' number of steps. The sequence diverges
    if the absolute value of z[i+1] is greater than 4.
    
    :param float x: the x component of the initial complex number
    :param float y: the y component of the initial complex number
    :param int threshold: the number of iterations to considered it converged
    """
    # initial conditions
    c = complex(x, y)
    z = complex(0, 0)
    
    for i in range(threshold):
        z = z**2 + c
        if abs(z) > 4.:  # it diverged
            return i
        
    return threshold - 1  # it didn't diverge

def animate(i):
    ax.clear()  # clear axes object
    #ax.set_xticks([], [])  # clear x-axis ticks
    #ax.set_yticks([], [])  # clear y-axis ticks
    
    X = np.empty((len(re), len(im)))  # re-initialize the array-like image
    threshold = round(1.15**(i + 1))  # calculate the current threshold
    
    # iterations for the current threshold
    for i in range(len(re)):
        for j in range(len(im)):
            X[i, j] = mandelbrot(re[i], im[j], threshold)
    
    # associate colors to the iterations with an iterpolation
    img = ax.imshow(X.T, interpolation="bicubic", cmap='magma')
    return [img]

def transformation(x,parameters):
     A = np.array([[parameters[0],parameters[1]],[parameters[2],parameters[3]]])
     b = np.array([parameters[4],parameters[5]])

     f = A.dot(x) + b

     return f

def transformation_alternative(x,r,s,a,e=0,f=0):
     A = np.array([[r*np.cos(a),-s*np.sin(a)],[r*np.sin(a),s*np.cos(a)]])
     b = np.array([e,f])

     f = A.dot(x) + b

     return f


#General transformations
def stretch(x,k):
    return transformation_alternative(x,1,k,0)

def squeeze(x,k):
    return transformation_alternative(x,k,1/k,0)

def rotation(x,angle):
    return transformation_alternative(x,1,1,angle)


#IFS-parameters, given in the form [a,b,c,d,e,f,p] where
#(a b)(x) + (e)
#(c d)(y) + (f)
#and p = probalilty
barnsley_fern = [[0,0,0,0.16,0,0,0.01],
                 [0.85,0.04,-0.04,0.85,0,1.6,0.85],
                 [0.2,-0.26,0.23,0.22,0,1.6,0.07],
                 [-0.15,0.28,0.26,0.24,0,0.44,0.07]]

fractal_tree = [[0,0,0,0.5,0,0,0.05],
                [0.42,-0.42,0.42,0.42,0,0.2,0.4],
                [0.42,0.42,-0.42,0.42,0,0.2,0.4],
                [0.1,0,0,0.1,0,0.2,0.15]]

square_fractal = [[0.5,0,0,0.5,1,1,0.25],
                  [0.5,0,0,0.5,50,1,0.25],
                  [0.5,0,0,0.5,1,50,0.25],
                  [0.5,0,0,0.5,50,50,0.25]]

Sierpinski_triangle = [[0.5,0,0,0.5,1,1,0.33],
                       [0.5,0,0,0.5,1,50,0.33],
                       [0.5,0,0,0.5,50,50,0.34]]

lewy_dragon = [[0.5,-0.5,0.5,0.5,0,1,0.5],
               [-0.5,-0.5,0.5,-0.5,0,0,0.5]]

golden_dragon = [[0.62,-0.4,0.4,0.62,0,0,0.5],
               [-0.38,-0.4,0.4,-0.38,1,0,0.5]]

Sierpinski_carpet = [[1/3,0,0,1/3,0,0,1/8],
                     [1/3,0,0,1/3,0,1/3,1/8],
                     [1/3,0,0,1/3,0,2/3,1/8],
                     [1/3,0,0,1/3,1/3,0,1/8],
                     [1/3,0,0,1/3,1/3,2/3,1/8],
                     [1/3,0,0,1/3,2/3,0,1/8],
                     [1/3,0,0,1/3,2/3,1/3,1/8],
                     [1/3,0,0,1/3,2/3,2/3,1/8]]

maple_leaf = [[0.14, 0.01, 0, 0.51,-0.08,-1.31, 0.25],
              [0.43, 0.52, -0.45, 0.5, 1.49,-0.75, 0.25],
              [0.45, -0.49, 0.47, 0.47, -1.62,-0.74, 0.25],
              [0.49, 0, 0, 0.51, 0.02, 1.62, 0.25]]

def IFS(parameters,iterations):
    v = np.array([0,0])
    x = []
    y = []

    w = [i[0:6] for i in parameters]
    p = [i[-1] for i in parameters]
    n = [i for i in range(len(p))]
    
    for i in range(iterations):
        r = np.random.choice(n, p=p)
        
        v = transformation(v,w[r])
        
        x.append(v[0])
        y.append(v[1])

    plt.scatter(x, y, s = 0.2, edgecolor ='green') 
    plt.show()     


test = [[random(),random(),random(),random(),random(),random(),0.25],
        [random(),random(),random(),random(),random(),random(),0.25],
        [random(),random(),random(),random(),random(),random(),0.25],
        [random(),random(),random(),random(),random(),random(),0.25]]

#IFS(fractal_tree,50000)

def numOfArrays(n: int, m: int, k: int):
    total_sum = 0
    M = 10**9+7
    for x in range(1,m-k+2):
        total_sum += (pow(x,n-1,mod=M)+M)%M
        
    return total_sum%M

print([[0] * (3) for _ in range(3)])

'''
X = np.empty((len(re), len(im)))
for i in range(len(re)):
        for j in range(len(im)):
            X[i, j] = mandelbrot(re[i], im[j], 50)

img = ax.imshow(X.T, interpolation="bicubic", cmap='magma')
plt.show()
'''