# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as mp
import resolution as res

# For all the function a0,b0 and c0 are 4 dimentional vectors representing (x_position,y_position,x_speed,y_speed)

def two_body_problem(a0,b0,mass_a,N,h) :    
    f = np.array([lambda t, yt : yt[2],
                         lambda t, yt : yt[3],
                         lambda t, yt : mass_a * (a0[0] - yt[0]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5,
                         lambda t, yt : mass_a * (a0[1] - yt[1]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5])
    sol = res.meth_n_step_printer(b0, 0, N, h, f, res.step_kutta)
    return sol
# For plotting
    # mp.plot(sol[:,0],sol[:,1])
    # mp.scatter(a0[0],a0[1],c="g",s=100)
    # mp.scatter(b0[0],b0[1],c="b",s=20)
    # mp.show()
    

#here b0 = (cos(t),sin(t))
def three_body_problem(a0,c0,b0,mass_a,mass_b,N,h):    
    f = np.array([lambda t, yt : yt[2],
                         lambda t, yt : yt[3],
                         lambda t, yt : mass_a * (a0[0] - yt[0]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5
                         +
                         mass_b * (b0[0](t) - yt[0]) / ((b0[0](t) - yt[0])**2 + (b0[1](t) - yt[1])**2)**1.5,
                         lambda t, yt : mass_a * (a0[1] - yt[1]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5
                         +
                         mass_b * (b0[1](t) - yt[1]) / ((b0[0](t) - yt[0])**2 + (b0[1](t) - yt[1])**2)**1.5])
     
    sol =  res.meth_n_step_printer(c0, 0, N, h, f, res.step_euler)
    return sol
# For plotting
    # mp.plot(sol[:,0],sol[:,1])
    # mp.scatter(a0[0],a0[1],c="g",s=100)
    # mp.scatter(c0[0],c0[1],'b', s= 10) 
    # mp.show()
def three_body_problem_base_change(a0,c0,b0_temps,masse_a,masse_b,N,h):
    c0_deriv = np.array([lambda t, yt : yt[2],
                         lambda t, yt : yt[3],
                         lambda t, yt : masse_a * (a0[0] - yt[0]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5
                         +
                         masse_b * (b0_temps[0](t) - yt[0]) / ((b0_temps[0](t) - yt[0])**2 + (b0_temps[1](t) - yt[1])**2)**1.5,
                         lambda t, yt : masse_a * (a0[1] - yt[1]) / ((a0[0] - yt[0])**2 + (a0[1] - yt[1])**2)**1.5
                         +
                         masse_b * (b0_temps[1](t) - yt[1]) / ((b0_temps[0](t) - yt[0])**2 + (b0_temps[1](t) - yt[1])**2)**1.5])
        
    tab =  res.meth_n_step_printer(c0, 0, N, h, c0_deriv, res.step_euler)
    x = np.zeros(len(tab[:,0]))
    y = np.zeros(len(tab[:,0]))
    t = 0
    for i in range (0,tab[:,0].size):
        x[i] = tab[i,0]*np.cos(i*h)+tab[i,1]*np.sin(i*h)
        y[i] = -tab[i,0]*np.sin(i*h)+tab[i,1]*np.cos(i*h)

    return (x,y)

# a0 = np.array([0,0,0,0])
# b0 = np.array([1,0,1,1])
# mass_a = 100
# N = 50
# h = 0.01
# two_body_problem(a0,b0,mass_a,N,h)
