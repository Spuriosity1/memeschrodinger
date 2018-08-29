import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse


parser = argparse.ArgumentParser(description='Simulate some q u a n t u m')

parser.add_argument('-e', nargs=1, type=int,
    help='log2 of the number of cells to brreak the interval into')
parser.add_argument('--states', nargs=2, type=int,
    help='The minimum and maximum indices to find')
parser.add_argument('--noplot', action='store_true', help='Do not save any results')
parser.add_argument('--pref', nargs=1,
    help='File prefix for saving.')
parser.add_argument('--scale', nargs=1, type=float,
        help='Potential scaling.')
parser.add_argument('--meme', nargs=1,
        help='meme to feed in.')

args=parser.parse_args()

'''
Units:
Everything in ???
'''

hb = 1
m=1

N=2**(args.e[0] if args.e else 10)
interval = args.states if args.states else [0,5]
scale = args.scale[0] if args.scale else 1

def Potential(x,y):
    #return np.zeros_like(x)
    return scale*((x-Lx/2)**2 + (y-Ly/2)**2)

Lx=1
Ly=1



print("Finding eigenvalues {} to {}".format(interval[0],interval[1]))

dx=Lx/N
dy=Ly/N

rx=1/(dx**2)
ry=1/(dy**2)


# H |psi> = E |psi>

X = np.linspace(0,Lx,N).repeat(N).reshape((N,N)).T
Y = np.linspace(0,Ly,N).repeat(N).reshape((N,N))
print(X,Y)
# laplacian = (skew-2*np.eye(N))/(delta**2)

laplacian = np.zeros((N**2,N**2))
for i in range(1,N-1):
    for j in range(1,N-1):
        laplacian[i+N*j,i+N*j]=2*rx+2*ry
        laplacian[i+1+N*j,i+N*j]=-rx
        laplacian[i-1+N*j,i+N*j]=-rx
        laplacian[i+N*(j-1),i+N*j]=-ry
        laplacian[i+N*(j+1),i+N*j]=-ry


V= Potential(X,Y)

print('V:\n',V)

H = -(hb**2/(2*m))*laplacian + np.diag(np.reshape(V,N*N))


# Where the magic happens

E, psi = linalg.eigh(H,eigvals=tuple(interval))
print("E numerical:\n",E)

if not(args.noplot):
    fig, ax1 = plt.subplots()
    ax1.set_label('Space')
    func=psi[:,0]

    func=func.reshape((N,N))
    print(func)
    im=ax1.imshow(func**2)
    fig.colorbar(im)
    fname = 'img2/'+args.pref[0] if args.pref else 'img/plot'

    fname +='-{}x{}-N{}-S{}-scale{}'.format(Lx,Ly,N,interval,round(scale))

    plt.savefig(fname)
    print('Plot saved as ""'+fname+'"')

    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    im=ax1.plot_surface(X,Y,V)

    fname = 'img2/_potential'+args.pref[0] if args.pref else 'img/pot-plot'

    fname +='-{}x{}-N{}-scale{}'.format(Lx,Ly,N,round(scale))

    plt.savefig(fname)
    print('Potential saved as ""'+fname+'"')
