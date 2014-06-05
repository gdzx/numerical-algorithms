# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as mp

FIRST_N = 50
MAX_STEP = 50

#######################
#  Resolution methods #
#######################
# conventions:
# ------------
# y is an array of values
# f is an array of functions
#######################

# step_<name>(y,t,h,f) : calculates one step

def step_euler(y, t, h, f):
    """ Euler method  """
    n = y.size
    yn = range(0, n)
    for i in range(0, n):
        yn[i] = y[i] + h * f[i](t, y)
    return np.array(yn)

def step_middle(y, t, h, f):
    """ Middle method  """
    n = y.size
    yn = range(0, n)
    y1 = range(0, n)
    step = range(0, n)
    for i in range(0, n):
        y1[i] = y[i]+(h/2.)*f[i](t,y)
        step[i] = f[i](t+h/2., y1)
        yn[i] = y[i] + h*step[i]
    return np.array(yn)

# Heun
def step_heun(y, t, h, f):
    """ Heun method  """
    n = y.size
    yn = range(0, n)
    step1 = range(0, n)
    step2 = range(0, n)
    y1= range(0, n)
    for i in range(0, n):
        step1[i] = f[i](t,y)
        y1[i] = y[i]+h*step1[i]
        step2[i] = f[i](t+h,y1)
        yn[i] = y[i]+(h/2.)*(step1[i]+step2[i])
    return np.array(yn)

# Runge-Kutta
def step_kutta(y, t, h, f):
    """ Runge-Kutta method """
    n = y.size
    yn = range(0, n)
    step1 = range(0, n)
    step2 = range(0, n)
    step3 = range(0, n)
    step4 = range(0, n)
    y1= range(0, n)
    y2= range(0, n)
    y3= range(0, n)
    for i in range(0, n):
        step1[i] = f[i](t,y)
        y1[i] = y[i]+ 0.5*h*step1[i]
        step2[i] = f[i](t+0.5*h,y1)
        y2[i] = y[i]+0.5*h*step2[i]
        step3[i] = f[i](t+0.5*h,y2)
        y3[i] = y[i]+h*step3[i]
        step4[i] = f[i](t+h,y3)
        yn[i] = y[i] + (1/6.)*h*(step1[i]+2*step2[i]+2*step3[i]+step4[i])
    return np.array(yn)

# meth_n_step(y0,t0,N,h,f,meth) : calculates N steps of size h #
def meth_n_step(y0, t0, N, h, f, meth):
    """ Compute n values """
    for i in range(0, N):
        y0 = meth(y0, t0, h, f)
        t0 = t0 + h
    return y0

def meth_n_step_printer(y0, t0, N, h, f, meth):
    """ Return an array of values """
    y_tab = np.zeros([N,y0.size])
    for i in range(0,N):
        y0 = meth(y0, t0, h, f)
        t0 = t0 + h
        y_tab[i] = y0
    return y_tab

# Solution with an error parameter eps
def meth_epsilon(y0, t0, tf, eps, f, meth):
    """ Returns an array of values """
    flag = 0
    error = eps + 1
    N = first_n
    h = (tf-t0) / float(N)
    yf_old = meth_n_step(y0, t0, N, h, f, meth)
    while (error > eps and flag < MAX_STEP):
        N *= 2
        h /= 2
        yf = meth_n_step(y0, t0, N, h, f, meth)
        error = np.linalg.norm(yf - yf_old)
        yf_old = yf
        flag += 1
    if flag == MAX_STEP:
        print("More steps are needed")
    return yf


def meth_epsilon_printer(y0, t0, tf, eps, f, meth, first_n = 2):
    flag = 0
    error = eps + 1
    N = first_n
    h = (tf-t0) / float(N)
    yf_old = meth_n_step(y0, t0, N, h, f, meth)
    while (error > eps and flag < MAX_STEP):
        N *= 2
        h /= 2
        yf = meth_n_step(y0, t0, N, h, f, meth)
        error = np.linalg.norm(yf - yf_old)
        yf_old = yf
        flag += 1
    y_tab = meth_n_step_printer(y0, t0, N, h, f, meth)
    if flag == MAX_STEP:
        print("More steps are needed")
    return y_tab

# slope field

def slope_field (deriv_x, deriv_y, n_x, n_y):
        h = 0.5
        X = np.arange(-n_x/2, n_x/2+1, h)
        Y = np.arange(-n_y/2, n_y/2+1, h)
        for i in X:
            for j in Y:
                mp.quiver(i, j, deriv_x(i, j),deriv_y(i, j))
        mp.show()
