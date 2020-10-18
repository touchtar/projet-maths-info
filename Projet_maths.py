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

# Fonction Newton qui écrit pas à pas dans un fichier. Peut s'avérer utile pour tracer des graphes.
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

#Level curve fonctionne. Problème: la fonction rebondit aux croisements et y fait des aller retours.
def level_curve(f,f_cont, x, y, N=150, eps=0.01):
    tab = np.zeros(shape = (2,N), dtype = float)
    tab[0,0] = x
    x_start = x
    tab[1,0] = y
    y_start = y
    global x0 
    global y0
    for i in range(1,N):
        grad_f = grad(f)
        g = grad_f(x,y)
        g_norm = delta*g/np.linalg.norm(g)
        J_f = J(f_cont)
        x,y = Newton(f_cont,x+g_norm[1],y-g_norm[0],eps,N)
        # J'empêche la fonction de faire plusieurs tours de courbe de niveau.
        if abs(x-x_start) < delta/2 and abs(y-y_start) < delta/2:
            break_index = i
            tab = tab[:,:i]
            break
        tab[0,i] = x
        tab[1,i] = y
        # Déclarer les points obetnus en global permet de les garder facilement pour la boucle suivante,
        # pour calculer la contrainte du cercle sans prise de tête avec les arguments qui posent problème
        # dans le calcul du jacobien.
        x0,y0 = x,y
    return tab

# J'implémente la contrainte pour que le point reste sur le cercle de centre le point précédent de rayon delta. La méthode de Newton
# cherchera donc un point exclusivement sur ce cercle. Garantit la distance entre les points de la courbe de niveau.
def contrainte_cercle(x,y):
    return (x-x0)**2+(y-y0)**2-delta**2

# Fonctions de référence.
def f1(x1,x2):
    x1 = np.array(x1)
    x2 = np.array(x2)
    return 3.0*x1*x1-2.0*x1*x2+3.0*x2*x2

def f2(x1, x2):
    return (x1 - 1)**2 + (x1 - x2**2)**2

def f3(x1, x2):
    return np.sin(x1 + x2) - np.cos(x1 * x2) - 1 + 0.001 * (x1 * x1 + x2 * x2)
 
# Fonctions modifiées avec la contrainte première bissectrice.
def f1_start(x1,x2):
    return np.array([f1(x1,x2)-c,x1-x2])

def f2_start(x1,x2):
    return np.array([f2(x1,x2)-c,x1-x2])

def f3_start(x1,x2):
    return np.array([f3(x1,x2)-c,x1-x2])

# Fonctions étendues avec la contrainte du cercle.
def f1_cont(x,y):
    return np.array([f1(x,y)-c,contrainte_cercle(x,y)])

def f2_cont(x,y):
    return np.array([f2(x,y)-c,contrainte_cercle(x,y)])

def f3_cont(x,y):
    return np.array([f3(x,y)-c,contrainte_cercle(x,y)])

# Dessin courbes de niveau et points calculés mis dans un array (2,N).
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

# Déclaration du point de départ de la recherche de courbe de niveau et des différents paramètres utiles au programme.
# J'ai passés la valeur c de la ligne de niveau et l'écart delta en global pour plus de clarté dans mon code et éviter les arguments
# récurrents et inutiles.
x = 2.0
y = 2.0
eps = 0.01
N = 100
global c
global delta
c = -2.4
delta = 0.1

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
x0,y0 = Newton(f3_start,x,y,eps,N)
tableau = level_curve(f3,f3_cont,x0,y0)
display_contour(f3,x=np.linspace(-5.0, 5.0, 100),y=np.linspace(-5.0, 5.0, 100),levels=5 ,array=tableau,title="Méthode de Newton pour les points (0.8,0.8) et (-1.0,-1.0)"+"\n"+"Fonction quadratique")