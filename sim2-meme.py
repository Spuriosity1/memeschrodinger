import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import sys
import argparse
import cv2


parser = argparse.ArgumentParser(description='Simulate some q u a n t u m')

parser.add_argument('--states', nargs=2, type=int,
    help='The minimum and maximum indices to find')
parser.add_argument('--noplot', action='store_true', help='Do not save any results')
parser.add_argument('-v', action='store_true', help='Plot all shit computed')
parser.add_argument('--invert', action='store_true', help='Invert the picture')
parser.add_argument('--pref', nargs=1,
    help='File prefix for saving.')
parser.add_argument('--scale', nargs=1, type=float,
        help='log2 of Potential scaling.')
parser.add_argument('--meme', nargs=1,
        help='meme to feed in.')

args=parser.parse_args()

if not args.meme:
    print("Wow, such empty. Please specify a meme.")
    exit()

meme = cv2.imread(args.meme[0])[:,:,0]
if args.invert:
    meme = -meme

'''
Units:
Everything in ???
'''

hb = 1
m=1

interval = args.states if args.states else [0,5]
scale = 10**args.scale[0] if args.scale else 1




print("Finding eigenvalues {} to {}".format(interval[0],interval[1]))

# Nx=meme.shape[0]
# Ny=meme.shape[1]

Nx,Ny=np.shape(meme)

print("This is a {} by {} image".format(Nx,Ny))

Nmax = Nx if Nx>Ny else Ny
rx=1/(Nx**2)
ry=1/(Ny**2)


# H |psi> = E |psi>

X = np.linspace(0,Nx,Nx).repeat(Ny).reshape((Ny,Nx)).T
Y = np.linspace(0,Ny,Ny).repeat(Nx).reshape((Nx,Ny))

# laplacian = (skew-2*np.eye(N))/(delta**2)
# I squished 2 dimensions into 1, lmao
laplacian = np.zeros(( Nx*Ny, Nx*Ny))
for i in range(1,Nx-1):
    for j in range(1,Ny-1):
        laplacian[Ny*i+j,Ny*i+j]=2*rx+2*ry
        laplacian[Ny*(i+1) + j,   Ny*i + j]=-rx
        laplacian[Ny*(i-1) + j,   Ny*i + j]=-rx
        laplacian[Ny*i     + j-1, Ny*i + j]=-ry
        laplacian[Ny*i     + j+1, Ny*i + j]=-ry


V=scale*np.array(meme)

print('V:\n',V)

H = -(hb**2/(2*m))*laplacian + np.diag(np.reshape(V, Nx*Ny))


# Where the magic happens

E, psi = linalg.eigh(H,eigvals=tuple(interval))
print("E numerical:\n",E)

if not(args.noplot):
    sum = np.sum(psi**2,axis=1)

    cmap = plt.get_cmap('PiYG')

    if args.v:
        for i in range(interval[1]-interval[0]+1):
            fig, ax1 = plt.subplots()
            ax1.set_label('Space')
            func=psi[:,i]

            func=func.reshape((Nx,Ny))
            im=ax1.imshow(func**2)
            fig.colorbar(im)
            fname = 'img2/'+args.pref[0] if args.pref else 'img/plot'

            imgsrc=args.meme[0].replace('.','_').replace('/','')
            fname +='-src_{}-N{}-scale{}'.format(imgsrc,i+interval[0],args.scale[0]).replace('.','_').replace('/','')

            plt.savefig(fname)
            print('Plot saved as ""'+fname+'"')



    sum=sum.reshape((Nx,Ny))
    levels = MaxNLocator(nbins=15).tick_values(sum.min(), sum.max())

    ########### FOR 3D PLOT
    fig = plt.figure()
    ax = Axes3D(fig)
    cf=ax.plot_surface(X,Y,np.flip(sum,axis=0),cmap=cm.viridis)

    ########## FOR CONTOUR
    # fig, ax1=plt.subplots()
    # cf=ax1.pcolormesh(np.flip(sum,axis=0))
    # cf = ax1.contourf(X,Y, np.flip(sum,axis=0), levels=levels,cmap=cm.viridis)
    #
    fig.colorbar(cf)

    fname = 'img2/'+args.pref[0] if args.pref else 'img/plot'

    imgsrc=args.meme[0]
    fname +='-src_{}-SUM{}-scale{}'.format(imgsrc,interval,args.scale[0]).replace('.','_').replace('/','')
    plt.show()
    plt.title('When you see a repost but it\'s the average of the first {}\n energy eigenstates of a Senate-shaped potential well'.format(interval[1]))
    plt.savefig(fname)
    print('Plot saved as ""'+fname+'"')



    # Plot the potential
    fig, ax1 = plt.subplots()
    im=ax1.imshow(V)
    fig.colorbar(im)


    fname = 'img2/_potential/'+args.pref[0] if args.pref else 'img/pot-plot'

    fname +='-src_{}-scale{}'.format(imgsrc,args.scale[0]).replace('.','_').replace('/','')

    plt.savefig(fname)
    print('Potential saved as ""'+fname+'"')
