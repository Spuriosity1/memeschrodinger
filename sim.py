import numpy as np
import scipy as sp
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import sys
import argparse


parser = argparse.ArgumentParser(description='Simulate some q u a n t u m')

parser.add_argument('-e', nargs=1, type=int,
    help='log2 of the number of cells to brreak the interval into')
parser.add_argument('--states', nargs=2, type=int,
    help='The minimum and maximum indices to find')
parser.add_argument('--noplot', action='store_true', help='Do not save any results')
parser.add_argument('--potential', action='store_true', help='Graph potential as well')
parser.add_argument('--pref', nargs=1,
    help='File prefix for saving.')
parser.add_argument('-s',nargs=1,type=float,help='Scaling of potential well')
parser.add_argument('--Escale',nargs=2,type=float,help='Energy bounds for potential well')

args=parser.parse_args()

'''
Units:
Everything in eV
'''

hb = 4.135667e-15
m=510998 #eV/c^2
omega = 1e-14/hb
L=1e-9

Echar=np.pi*hb*np.pi*hb/(2*m*L*L)

if not args.s:
    args.s = 1

print("Characteristic energy: ", Echar, 'eV')
print()

def V(x):
    s=Echar*(args.s[0] if args.s else 1)
    #return np.zeros_like(x)

    # z= -s*np.abs(2*x/L-1)

    # z = s*np.power((2*x/L-1),3)

    z = s*(4*x/L-1)**2*(4*x/L-3)**2

    # z= np.zeros_like(x)
    # z[2*N//5:N*3//5]=s


    return z



N=2**(args.e[0] if args.e else 10)
interval = args.states if args.states else [0,5]

print("Finding eigenvalues {} to {}".format(interval[0],interval[1]))

delta=L/N

X = np.linspace(0,L,N)

# H |psi> = E |psi>
skew = np.roll(np.eye(N),1,axis=1)
skew[:,0]=np.zeros(N).T

skew2 = np.roll(skew,1,axis=1)
skew2[:,0]=np.zeros(N).T

skew += skew.T
skew2 += skew2.T

#laplacian = (skew-2*np.eye(N))/(delta**2)

laplacian = (16*skew - skew2 -30*np.eye(N))/(12*delta**2)
# laplacian[0,:] = np.zeros(N)
# laplacian[-1,:] = np.zeros(N)
# laplacian[:,0] = np.zeros(N).T
# laplacian[:,-1] = np.zeros(N).T

H = -(hb**2/(2*m))*laplacian + np.diag(V(X))



# Where the magic happens

E, psi = eigh(H,eigvals=tuple(interval))

# normalise
integrals=np.power(np.sum(np.power(psi,2),axis=0),-0.5)*L

psi = integrals*psi

print("E numerical:\n",E)

if not(args.noplot):
    fig, ax1 = plt.subplots()
    ax1.set_label('Space')
    for i in range(interval[1]-interval[0]+1):
        ax1.plot(X, psi[:,i]**2, label='n={}'.format(interval[0]+i))
    ax1.legend()

    if args.potential:
        ax2=ax1.twinx()
        ax2.plot(X,V(X),'b--',linewidth=1.0)

        if args.Escale:
             ax2.set_ylim(args.Escale[0]*Echar,args.Escale[1]*Echar)

        ax2.set_ylabel('V (eV)')

    fname = 'img/'+args.pref[0] if args.pref else 'img/plot'

    fname +='-L{}-N{}-vals{}-S{}.png'.format(L,N,interval,args.s)

    plt.savefig(fname)
    print('Plot saved as ""'+fname+'"')
