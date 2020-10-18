import autograd
import autograd.numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]
from IPython.display import display

def grad(f):
    g = autograd.grad
    def grad_f(x, y):
        return np.array([g(f, 0)(x, y), g(f, 1)(x, y)])
    return grad_f

def J(f):
    j = autograd.jacobian
    def J_f(x, y):
        return np.array([j(f, 0)(x, y), j(f, 1)(x, y)]).T
    return J_f

def Newton(F, x0, y0, eps, N):
    J_f = J(F)
    v0 = np.array([x0,y0])
    for i in range(N):
        v0 = v0 - np.dot(np.linalg.inv(J_f(v0[0],v0[1])),F(v0[0],v0[1]))
        x,y = v0[0],v0[1]
        if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
            return x, y
        x0, y0 = x, y
    else:
        raise ValueError(f"no convergence in {N} steps.")

def Newton_steps(F, x0, y0, eps, N,file):
    with open(file,"a") as data:
        J_f = J(F)
        v0 = np.array([x0,y0])
        data.write(str(x0)+" "+str(y0)+"\n")
        for i in range(N):
            v0 = v0 - np.dot(np.linalg.inv(J_f(v0[0],v0[1])),F(v0[0],v0[1]))
            x,y = v0[0],v0[1]
            data.write(str(x)+" "+str(y)+"\n")
            if np.sqrt((x - x0)**2 + (y - y0)**2) <= eps:
                return x, y
            x0, y0 = x, y
        else:
            raise ValueError(f"no convergence in {N} steps.")

#Level curve fonctionne, reste à l'empêcher de faire plusieurs tours. Problème, la fonction rebondit aux croisements et fait des aller retours  
def level_curve(f,f_cont, x, y, delta=0.1, N=150, eps=0.01):
    tab = np.zeros(shape = (2,N), dtype = float)
    tab[0,0] = x
    tab[1,0] = y
    for i in range(1,N):
        grad_f = grad(f)
        g = grad_f(x,y)
        g_norm = delta*g/np.linalg.norm(g)
        J_f = J(f_cont)
        x,y = Newton(f_cont,x+g_norm[1],y-g_norm[0],eps,N)
        tab[0,i] = x
        tab[1,i] = y
        global x0 
        global y0
        x0,y0 = x,y
    return tab


def contrainte_cercle(x,y,delta=0.1):
    return (x-x0)**2+(y-y0)**2-delta**2

def f1(x1,x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0*x1*x1-2.0*x1*x2+3.0*x2*x2

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

# J'implémente la contrainte pour que le point reste sur le cercle de centre le point précédent de rayon delta
def fbis(x,y,delta = 0.1,c=2.0):
    return np.array([f2(x,y)-c,contrainte_cercle(x,y,delta)])

# Fonction f1 modifiée avec la contrainte première bissectrice
def f(x1,x2,c=2.0):
    return np.array([f2(x1,x2)-c,x1-x2])

# Dessin courbes de niveau et points calculés mis dans un array
def display_contour(f, x, y, levels,array,title):
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    fig, ax = plt.subplots()
    contour_set = plt.contour(
        X, Y, Z, colors="grey", linestyles="dashed", 
        levels=levels 
    )
    ax.clabel(contour_set)
    plt.grid(True)
    plt.plot(array[0,:],array[1,:],'x')
    plt.xlabel("$x_1$") 
    plt.ylabel("$x_2$")
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.show()


x = 0.5
y = 0.5
eps = 0.001
N = 100
"""
#Analyse de l'écart entre la solution exacte et l'approximation en fonction du pt de départ
with open("Newton_test.txt","w") as data:
    a = 0.6
    while a < 1.0:
        x,y = a,a
        r = np.sqrt(0.2)
        res = Newton(f,x,y,eps,N)
        ecart = abs(res[0]-r)
        data.write(str(a)+" "+str(res[0])+" "+str(res[1])+" "+str(ecart)+"\n")
        a += 0.01

#Figure courbes de niveau à partir de deux points initiaux
list = [0.8,-1.0]
for i in list:
    Newton_steps(f,i,i,eps,N,"Newton_steps.txt")
display_contour(f1,x=np.linspace(-1.0, 1.0, 100),y=np.linspace(-1.0, 1.0, 100),levels=10,file ="Newton_steps.txt",title="Méthode de Newton pour les points (0.8,0.8) et (-1.0,-1.0)"+"\n"+"Fonction quadratique")
"""
#Détermination des courbes de niveau
x0,y0 = Newton(f,x,y,eps,N)
c = 0.8
tableau = level_curve(f1,fbis,x0,y0)
display_contour(f2,x=np.linspace(-1.0, 3.0, 100),y=np.linspace(-2.0, 2.0, 100),levels=[2**i for i in range(-3, 8)],array=tableau,title="Méthode de Newton pour les points (0.8,0.8) et (-1.0,-1.0)"+"\n"+"Fonction quadratique")