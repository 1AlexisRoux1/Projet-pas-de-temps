import numpy as np
import matplotlib.pyplot as plt
from math import *

def solve_euler_explicit_2(f,x0,dt,t0=0,tf=1):
    """
    Cette fonction résout une équation différentielle à l'aide de la méthode d'Euler et renvoie les tableaux des t et des x au cours de la résolution
    """
    s=floor(tf/dt)
    t = [k*dt for k in range(s+1)]
    x = np.zeros((s+1,2))
    x[0]=x0
    for k in range(0,s):
        x[k+1]=x[k]+dt*f(t[k],x[k])
    return t,x

def i(t,x):
    return np.array([x[1],-x[0]])

def h(t,x):
    return 3*x*(1-x/100)

def g(t,x):
    return np.array([x[1],-x[0]-x[1]])

def solve_rk_2_2(f,x0,dt,t0=0,tf=1):
    """
    Cette fonction résout une équation différentielle à l'aide de la méthode de Runge-Kata-2 et renvoie les tableaux des t et des x au cours de la résolution
    """
    s=floor(tf/dt)
    t = [k*dt for k in range(s+1)]
    x = np.zeros((s+1,2))
    x[0]=x0
    for k in range(0,s):
        x[k+1]=x[k]+dt*f(t[k]+dt/2,x[k]+dt/2*f(t[k],x[k]))
    return t,x

def affiche(courbe):
    for t,x in courbe:
        plt.plot(t,x)
    plt.show()



# te,xe=(solve_euler_explicit_2(i,[1,0],0.1,0,1000))
tr,xr=(solve_rk_2_2(i,[1,0],pi/10,0,500*pi))

ts=tr
xs=[]
for a in ts:
    xs.append(cos(a))
plt.plot(ts,xs,label='vraie solution',color='green')
# plt.plot(te,xe[::,0],label='solution déterminée avec le schéma dEuler',color='blue')
plt.plot(tr,xr[::,0],label='solution déterminée par RK2',color='red')
plt.legend()
plt.show()

cost=np.vectorize(cos)
abst=np.vectorize(abs)

# Calcul de l'erreur pour rk 2

tf=5*pi
dt=pi/20
x0=[1,0]
e=[]
for k in range(200):
    t,x=solve_rk_2_2(i,x0,dt,0,tf)
    finx =np.array(x[-10:,0])
    fint=np.array(t[-10:])
    vraisx=cost(fint)
    comparaison= abst(finx-vraisx)
    e.append([dt,max(comparaison)])
    dt=dt-pi/4000
e2=np.zeros((2,200))
for k in range(200):
    e2[0][k]=e[k][0]
    e2[1][k]=e[k][1]

plt.plot(e2[0],e2[1])
plt.show()

# Calcul de l'erreur pour Euler
tf=5*pi
dt=pi/20
x0=[1,0]
e=[]
for k in range(200):
    t,x=solve_euler_explicit_2(i,x0,dt,0,tf)
    finx =np.array(x[-10:,0])
    fint=np.array(t[-10:])
    vraisx=cost(fint)
    comparaison= abst(finx-vraisx)
    e.append([dt,max(comparaison)])
    dt=dt-pi/4000
e2=np.zeros((2,200))
for k in range(200):
    e2[0][k]=e[k][0]
    e2[1][k]=e[k][1]

plt.plot(e2[0],e2[1])

plt.show()



