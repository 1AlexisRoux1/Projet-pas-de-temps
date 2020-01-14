import numpy as np
import matplotlib.pyplot as plt
from math import *

def solve_euler_explicit(f,x0,dt,t0=0,tf=1):
    """
    Cette fonction résout une équation différentielle à l'aide de la méthode d'Euler et renvoie les tableaux des t et des x au cours de la résolution
    """
    s=floor(tf/dt)
    t = [k*dt for k in range(s+1)]
    x = np.zeros((s+1))
    x[0]=x0
    for k in range(0,s):
        x[k+1]=x[k]+dt*f(t[k],x[k])
    return t,x

def i(t,x):
    return -x

def h(t,x):
    return 3*x*(1-x/100)

def g(t,x):
    return np.array([x[1],-x[0]-x[1]])

def solve_rk_2(f,x0,dt,t0=0,tf=1):
    """
    Cette fonction résout une équation différentielle à l'aide de la méthode de Runge-Kata-2 et renvoie les tableaux des t et des x au cours de la résolution
    """
    s=floor(tf/dt)
    t = [k*dt for k in range(s+1)]
    x = np.zeros((s+1))
    x[0]=x0
    for k in range(0,s):
        x[k+1]=x[k]+dt*f(t[k]+dt/2,x[k]+dt/2*f(t[k],x[k]))
    return t,x

def affiche(courbe):
    for t,x in courbe:
        plt.plot(t,x)
    plt.show()



te,xe=(solve_euler_explicit(i,1,0.01))
tr,xr=(solve_rk_2(i,1,0.01))

ts=te
xs=[]
for a in ts:
    xs.append(exp(-a))
plt.plot(ts,xs,label='vraie solution',color='green')
plt.plot(te,xe,label='solution déterminée avec le schéma dEuler',color='blue')
plt.plot(tr,xr,label='solution déterminée par RK2',color='red')
plt.legend()
plt.show()























