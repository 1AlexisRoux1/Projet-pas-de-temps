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

def j(x):
    return np.array([x[1],-x[0]])


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



#te,xe=(solve_euler_explicit_2(i,[1,0],0.01,0,100))
tr,xr=(solve_rk_2_2(i,[1,0],pi/10,0,50*pi))

ts=tr
xs=[]
for a in ts:
    xs.append(cos(a))
plt.plot(ts,xs,label='vraie solution',color='green')
#plt.plot(te,xe[::,0],label='solution déterminée avec le schéma dEuler',color='blue')
plt.plot(tr,xr[::,0],label='solution déterminée par RK2',color='red')
plt.legend()
plt.show()

cost=np.vectorize(cos)
abst=np.vectorize(abs)
def square(x):
    return x*x
np.vectorize(square)
def cube(x):
    return x*x*x
np.vectorize(cube)
# Calcul de l'erreur pour rk 2

tf=5*pi
dt=pi/20
x0=[1,0]
r=[]
for k in range(200):
    t,x=solve_rk_2_2(i,x0,dt,0,tf)
    finx =np.array(x[-10:,0])
    fint=np.array(t[-10:])
    vraisx=cost(fint)
    comparaison= abst(finx-vraisx)
    r.append([dt,max(comparaison)])
    dt=dt-pi/4000
r2=np.zeros((2,200))
for k in range(200):
    r2[0][k]=r[k][0]
    r2[1][k]=r[k][1]

plt.plot(r2[0],r2[1],label='Véritable écart',green)
plt.plot(r2[0],(2.4*square(r2[0])),color='red',label='carré')
plt.plot(r2[0],(18*cube(r2[0])),color='green',label='cube')
plt.legend()
plt.xlabel('pas de temps')
plt.ylabel('écart entre solution calculée et solution réelle')
plt.title('Approximation de lerreur pour RK2 appliqué à un cosinus')
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

plt.plot(e2[0],e2[1],label='Véritable écart')
plt.plot(e2[0],15.5*e2[0],color='yellow',label='Linéaire')
plt.plot(e2[0],1000*square(e2[0]),color='red',label='carré')
plt.legend()
plt.xlabel('pas de temps')
plt.ylabel('écart entre solution calculée et solution réelle')
plt.title('Approximation de lerreur pour Euler appliqué à un cosinus')

plt.show()

def g(t,x):
    return np.array([x[1],-x[0]-x[1]/10])

tr,xr=(solve_rk_2_2(g,[1,0],pi/10,0,50*pi))
plt.plot(tr,xr[::,0],label='solution déterminée par RK2',color='red')
plt.legend()
plt.show()

def solve_ivp_euler_explicit_variable_step(f, t0, x0, t_f, dtmin = 1e-16, dtmax = 0.01, atol = 1e-6):
    dt = dtmax/10; # initial integration step
    ts, xs = [t0], [x0]  # storage variables
    t = t0
    ti = 0  # internal time keeping track of time since latest storage point : must remain below dtmax
    x = x0
    while ts[-1] < t_f:
        while ti < dtmax:
            t_next, ti_next, x_next = t + dt, ti + dt, x + dt * f(x)
            x_back = x_next - dt * f(x_next)
            ratio_abs_error = atol / (np.linalg.norm(x_back-x)/2)
            dt = 0.9 * dt * sqrt(ratio_abs_error)
            if dt < dtmin:
                raise ValueError("Time step below minimum")
            elif dt > dtmax/2:
                dt = dtmax/2
            t, ti, x = t_next, ti_next, x_next
        dt2DT = dtmax - ti # time left to dtmax
        t_next, ti_next, x_next = t + dt2DT, 0, x + dt2DT * f(x)
        ts = np.vstack((ts,[t_next]))
        xs = np.vstack((xs,[x_next]))
        t, ti, x = t_next, ti_next, x_next
    return (ts, xs.T)

tss,xss=solve_ivp_euler_explicit_variable_step(j,0,np.array([1,0]),50*pi,1e-16,pi/10)
tr,xr=(solve_rk_2_2(i,[1,0],pi/10,0,50*pi))
ts=tr
xs=[]
for a in ts:
    xs.append(cos(a))
plt.plot(ts,xs,label='vraie solution',color='green')
plt.plot(tr,xr[::,0],label='solution déterminée par RK2',color='red')
plt.plot(tss,xss[0],label='pas de temps adaptatif', color='yellow')
plt.legend()
plt.show()

# Calcul de l'erreur pour le pas adaptatif
tf=5*pi
dt=pi/20
x0=[1,0]
e=[]
for k in range(10):
    ts,xs=solve_ivp_euler_explicit_variable_step(j,0,[1,0],5*pi,1e-16,dt)
    vraisx=cost(ts)
    comparaison= abst(xs.T[0] - vraisx)
    e.append([dt,np.max(comparaison)])
    dt=dt-pi/200
e2=np.zeros((2,10))
for k in range(10):
    e2[0][k]=e[k][0]
    e2[1][k]=e[k][1]

plt.plot(e2[0],e2[1],label='Véritable écart')
#plt.plot(e2[0],15.5*e2[0],color='yellow',label='Linéaire')
#plt.plot(e2[0],1000*square(e2[0]),color='red',label='carré')
plt.legend()
plt.xlabel('pas de temps')
plt.ylabel('écart entre solution calculée et solution réelle')
plt.title('Approximation de lerreur pour Euler appliqué à un cosinus')
plt.show()

