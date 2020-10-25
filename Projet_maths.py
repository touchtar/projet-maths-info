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
def level_curve(f,f_cont, x, y,oversampling, N=150, eps=0.01):
    tab = np.zeros(shape = (2,N), dtype = float)
    grad_f = grad(f)
    tab[0,0] = x
    x_start = x
    tab[1,0] = y
    y_start = y
    g = grad_f(x,y)
    g_norm = delta*g/np.linalg.norm(g)
    g_norm_ant = g_norm
    global x0 
    global y0
    for i in range(1,N):
        g = grad_f(x,y)
        g_norm = delta*g/np.linalg.norm(g)
        J_f = J(f_cont)
        x,y = Newton(f_cont,x+g_norm[1],y-g_norm[0],eps,N)
        tab[0,i] = x
        tab[1,i] = y
        """
        if i != 1:
            if intersection_3([(tab[0,0],tab[1,0]),(tab[0,1],tab[1,1])],[(tab[0,i-1],tab[1,i-1]),(tab[0,i],tab[1,i])]):
                tab = tab[:,:i]
                return tab
        """
        if oversampling > 1:
            overtab = np.zeros(shape = (2,oversampling), dtype = float)
            k = 0
            for t in np.linspace(0.0,1.0,oversampling):
                gam = gamma(t,np.array([tab[0,i-1],tab[1,i-1]]),np.array([tab[0,i],tab[1,i]]),
                np.array([g_norm_ant[1],-g_norm_ant[0]]),np.array([g_norm[1],-g_norm[0]]))
                overtab[0,k] = gam[0]
                overtab[1,k] = gam[1]
                k += 1
            tab = np.column_stack([tab,overtab])
        g_norm_ant = g_norm       
        # Déclarer les points obetnus en global permet de les garder facilement pour la boucle suivante,
        # pour calculer la contrainte du cercle sans prise de tête avec les arguments qui posent problème
        # dans le calcul du jacobien.
        x0,y0 = x,y
    return tab

#Implémentation du chemin gamma(t).
def gamma(t,P1,P2,u1,u2):
    a = P1[0]
    d = P1[1]
    matrice = np.array([u1[0],u2[0],u1[1],u2[1]]).reshape((2,2))
    if np.linalg.det(matrice) != 0:
        lambmu = 2*np.dot(np.linalg.inv(matrice),(P2-P1))
        b = lambmu[0]*u1[0]
        e = lambmu[0]*u1[1]
        c = (lambmu[1]*u2[0]-lambmu[0]*u1[0])/2
        f = (lambmu[1]*u2[1]-lambmu[0]*u1[1])/2
        x = a+b*t+c*t**2
        y = d+e*t+f*t**2
    else:
        x = P1[0]+t*(P2[0]-P1[0])
        y = P1[1]+t*(P2[1]-P1[1])
    return np.array([x,y]).reshape((2,1))

# Intersection testée, ne fonctionne pas sur les segments qui se superposent.
def intersection_2(a1,a2,b1,b2):
    if a2[0]-a1[0] != 0:
        coef1 = (a2[1]-a1[1])/(a2[0]-a1[0])
        ord1 = a1[1]-a1[0]*coef1
    if b2[0]-b1[0] != 0:
        coef2 = (b2[1]-b1[1])/(b2[0]-b1[0])
        ord2 = b1[1]-b1[0]*coef1
    if coef1 == coef2 and ord1 != ord2:
        return False
    if coef1 == coef2 and ord2 == ord2:
        if (min(b1[0],b2[0]) < a1[0] < max(b1[0],b2[0]) or min(b1[0],b2[0]) < a2[0] < max(b1[0],b2[0]) 
        or min(a1[0],a2[0]) < b1[0] < max(a1[0],a2[0]) or min(a1[0],a2[0]) < b2[0] < max(a1[0],a2[0])):
            return True
        else:
            return False
    I1 = (ord2 - ord1) / (coef1 - coef2)
    if (I1 < max( min(a1[0],a2[0]), min(b1[0],b2[0]))) or (I1 > min( max(a1[0],a2[0]), max(b1[0],b2[0]))):
        return False
    else:
        return True

def intersection(a1,a2,b1,b2):
    K = ((a2[1]-a1[1])/(a2[0]-a1[0])-(b2[1]-b1[1])/(b2[0]-b1[0]))
    if K != 0:
        xi = (((a2[1]-a1[1])/(a2[0]-a1[0]))*a1[0]+a1[1]-((b2[1]-b1[1])/(b2[0]-b1[0]))*b1[0]-b1[1]) / K
    if (xi < max( min(a1[0],a2[0]), min(b1[0],b2[0]))) or (xi > min( max(a1[0],a2[0]), max(b1[0],b2[0]))):
        return False
    else:
        return True

def intersection_3(s0,s1):
    dx0 = s0[1][0]-s0[0][0]
    dx1 = s1[1][0]-s1[0][0]
    dy0 = s0[1][1]-s0[0][1]
    dy1 = s1[1][1]-s1[0][1]
    p0 = dy1*(s1[1][0]-s0[0][0]) - dx1*(s1[1][1]-s0[0][1])
    p1 = dy1*(s1[1][0]-s0[1][0]) - dx1*(s1[1][1]-s0[1][1])
    p2 = dy0*(s0[1][0]-s1[0][0]) - dx0*(s0[1][1]-s1[0][1])
    p3 = dy0*(s0[1][0]-s1[1][0]) - dx0*(s0[1][1]-s1[1][1])
    return (p0*p1<=0) & (p2*p3<=0)


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
x = 1.0
y = 1.0
eps = 0.01
N = 1000
global c
global delta
c = 1.5
delta = 0.05

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
display_contour(f1,x=np.linspace(-1.0, 1.0, 100),y=np.linspace(-1.0, 1.0, 100),
levels=10,file ="Newton_steps.txt",
title="Méthode de Newton pour les points (0.8,0.8) et (-1.0,-1.0)"+"\n"+"Fonction quadratique")
"""

#Détermination des courbes de niveau
x0,y0 = Newton(f1_start,x,y,eps,N)
tableau = level_curve(f1,f1_cont,x0,y0,10)
#print(tableau)
display_contour(f1,x=np.linspace(-1.0, 1.0, 100),y=np.linspace(-1.0, 1.0, 100),
levels=5 ,array=tableau,
title="Méthode de Newton pour les points (0.8,0.8) et (-1.0,-1.0)"+"\n"+"Fonction quadratique")

"""
# Représentation du fonctionnement de gamma
P1 = np.array([11.0,11.0])
P2 = np.array([15.0,24.0])
u1 = np.array([0.7,4.3])
u2 = np.array([1.0,0.6])
ax = plt.axes()
ax.arrow(P1[0],P1[1],u1[0],u1[1],width = 0.05)
ax.arrow(P2[0],P2[1],u2[0],u2[1],width = 0.05)
for t in np.linspace(0.0,1.0,100):
    gam = gamma(t,P1,P2,u1,u2)
    plt.scatter(gam[0],gam[1])
plt.show()
"""