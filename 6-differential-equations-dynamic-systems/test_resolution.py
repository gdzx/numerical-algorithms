# -*- coding: utf-8 -*-
import resolution as res
import numpy as np
import matplotlib.pyplot as mp
print
print "##########################################################"
print "                   TESTING RSOLUTION METHODS                         "
print "##########################################################"
print

# global variables
FIRST_N = 2
MAX_STEP = 50 

y0_dim1 = np.array([1])
dy_dim1 = np.array([lambda t, y : y[0]/(1+t**2)])
print "------ dimension = 1 ------"
print
print "These are the solutions given by all the methods for a cauchy system in dimention 1. The cauchy problem is defined in the report"
print
tab1 = res.meth_epsilon_printer(y0_dim1, 0, 1, 0.00001, dy_dim1, res.step_euler)
tab2 = res.meth_epsilon_printer(y0_dim1, 0, 1, 0.00001, dy_dim1, res.step_middle)
tab3 = res.meth_epsilon_printer(y0_dim1, 0, 1, 0.00001, dy_dim1, res.step_heun)
tab4 = res.meth_epsilon_printer(y0_dim1, 0, 1, 0.00001, dy_dim1, res.step_kutta)

x = np.arange(0, 1, 0.1)
x1 = np.arange(0 , 1 , 1.0 / tab1.size)
x2 = np.arange(0 , 1 , 1.0 / tab2.size)
x3 = np.arange(0 , 1 , 1.0 / tab3.size)
x4 = np.arange(0 , 1 , 1.0 / tab4.size)

euler, = mp.plot(x1,tab1[:,0],'b')
milieu, = mp.plot(x2,tab2[:,0],'r')
heun, = mp.plot(x3,tab3[:,0],'g')
kutta, = mp.plot(x4,tab4[:,0],'y')
exacte, = mp.plot(x, np.exp(np.arctan(x)),'o')


mp.legend((euler,milieu,heun,kutta,exacte),('Euler','Middle','Heun','Kutta','exact solution'),loc = 4)
mp.show()

print
print "------ dimension = 2 ------"
print
print
print "These are the solutions given by all the methods for a cauchy system in dimention 2. The cauchy problem is defined in the report"
print
y0_dim2 = np.array([1,0])
dy_dim2 = np.array([lambda t, y : -y[1],
                    lambda t, y : y[0]])

tab1 = res.meth_epsilon_printer(y0_dim2, 0, 1, 0.0001, dy_dim2, res.step_euler)
tab2 = res.meth_epsilon_printer(y0_dim2, 0, 1, 0.0001, dy_dim2, res.step_middle)
tab3 = res.meth_epsilon_printer(y0_dim2, 0, 1, 0.0001, dy_dim2, res.step_heun)
tab4 = res.meth_epsilon_printer(y0_dim2, 0, 1, 0.0001, dy_dim2, res.step_kutta)

# la solution exacte
x = np.arange(0, 3.14/2 ,0.1)
y = np.arange(0, 3.14/2 ,0.1)
for i in range (0,x.size):
    x[i] = np.cos(x[i])
    y[i] = np.sin(y[i])
    
    # traçage des courbes
euler, = mp.plot(tab1[:,0],tab1[:,1],'b')
middle, = mp.plot(tab2[:,0],tab2[:,1],'r')
heun, = mp.plot(tab3[:,0],tab3[:,1],'g')
kutta, = mp.plot(tab4[:,0],tab4[:,1],'y')
exacte, = mp.plot(x,y,'o')

mp.legend((euler,middle,heun,kutta,exacte),('Euler','Middle','Heun','Kutta','exact solution'),loc = 3)
mp.show()

print
print "---------------------------------------"
print
print "slope field dimension 1 ..."
print
deriv_x = lambda t, y: 1
deriv_y = lambda t, y: y/(1+t**2)
res.slope_field(deriv_x,deriv_y,7,7)
print
print "---------------------------------------"
print
print "slope field dimension 2 ..."
print
deriv_2x = lambda t, y: -deriv_y(t, y)
deriv_2y = lambda t, y: deriv_x(t, y)
res.slope_field(deriv_2x,deriv_2y,7,7)
 
