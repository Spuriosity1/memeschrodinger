import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse


parser = argparse.ArgumentParser(description='Simulate some q u a n t u m')

parser.add_argument('-e', nargs=1, type=int,
    help='the number of cells to break the interval into')
parser.add_argument('--nsols', nargs=1, type=int,
    help='the number of eigenvalues to look for')
parser.add_argument('--state', nargs=1, type=int,
    help='the eigenvalue to plot')
parser.add_argument('--noplot', action='store_true', help='Do not save any results')
parser.add_argument('--pref', nargs=1,
    help='File prefix for saving.')

args=parser.parse_args()

'''
Units:
Everything in ???
'''

hb = 1
m=1
omega = 15


def Potential(x,y):
    #return np.zeros_like(x)
    return 0.5*m*omega*omega*((x-Lx/2)**2 + (y-Ly/2)**2)

Lx=1
Ly=1

N=int(args.e[0] if args.e else 10)
nsols = args.nsols[0] if args.nsols else 10
state = args.state[0] if args.state else 0

dx=Lx/N
dy=Ly/N

rx=1/(dx**2)
ry=1/(dy**2)


# H |psi> = E |psi>

X = np.linspace(0,Lx,N).repeat(N).reshape((N,N)).T
Y = np.linspace(0,Ly,N).repeat(N).reshape((N,N))
print(X,Y)
# laplacian = (skew-2*np.eye(N))/(delta**2)

NUMBYTES = N*N*N*N*8

if NUMBYTES > 1e9:
    print("Now creating a {} byte matrix. Do you really want to do this?".format(NUMBYTES))
    ans=input()
    if (ans[0].lower() == 'n'):
        exit()

laplacian = np.zeros((N**2,N**2))
for i in range(1,N-1):
    for j in range(1,N-1):
        laplacian[i+N*j,i+N*j]=2*rx+2*ry
        laplacian[i+1+N*j,i+N*j]=-rx
        laplacian[i-1+N*j,i+N*j]=-rx
        laplacian[i+N*(j-1),i+N*j]=-ry
        laplacian[i+N*(j+1),i+N*j]=-ry

laplacian = sparse.dia_matrix(laplacian)

V= Potential(X,Y)

print('V:\n',V)

H = -(hb**2/(2*m))*laplacian + sparse.dia_matrix(np.diag(np.reshape(V,N*N)))


# Where the magic happens

E, psi = linalg.eigsh(H,k=nsols)
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

    fname +='-{}x{}-N{}-S{}{}'.format(Lx,Ly,N,nsols,state)

    plt.savefig(fname)
    print('Plot saved as ""'+fname+'"')

    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    im=ax1.plot_surface(X,Y,V)

    fname = 'img2/_potential'+args.pref[0] if args.pref else 'img/pot-plot'

    fname +='-{}x{}-N{}'.format(Lx,Ly,N)

    plt.savefig(fname)
    print('Potential saved as ""'+fname+'"')
