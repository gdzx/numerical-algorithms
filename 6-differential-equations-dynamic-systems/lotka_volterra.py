# -*- coding: utf-8 -*-

# Packages
import numpy as np
import matplotlib.pyplot as mp
from resolution import *

class malthus:
    def __init__(self, n0, gamma):
        self.equation = self.make_equation(n0, gamma)

    def make_equation(self, n0, gamma):
        return np.array([n0]), np.array([lambda t, N: gamma*N[0]])

    def __str__(self):
        return "Malthus"

class verhulst:
    def __init__(self, n0, gamma, kappa):
        self.equation = self.make_equation(n0, gamma, kappa)

    def make_equation(self, n0, gamma, kappa):
        return np.array([n0]), np.array([lambda t, N: gamma*N[0]*(1-N[0]/kappa)])

    def __str__(self):
        return "Verhulst"

class lotka_volterra:
    def __init__(self, n0, m0, a, b, c, d):
        self.equation = self.make_equation(n0, m0, a, b, c, d)

    def make_equation(self, n0, m0, a, b, c, d):
        return np.array([n0, m0]), np.array([lambda t, y : y[0]*(a-b*y[1]),
                                             lambda t, y : y[1]*(c*y[0]-d)])

    def __str__(self):
        return "Lotka-Volterra"

def graph_time(obj, step, t0 = 0, tf = 5, epsilon = 0.1, first_n = None, title = ""):
    y0, dy = obj.equation
    if not first_n:
        first_n = 4*(tf-t0)
    tab = meth_epsilon_printer(y0, t0, tf, epsilon, dy, step, first_n)
    x = np.linspace(t0, tf, len(tab[:,0]))
    mp.plot(x, tab[:,0], 'b')
    if (tab.shape[1] == 2):
        mp.plot(x, tab[:,1], 'r')
        mp.legend(["Proies", "Predateurs"])
    mp.xlabel("Temps")
    mp.ylabel("Population")
    mp.title(title)
    mp.show()
    mp.clf()

def trajectory(obj, step, t0 = 0, tf = 5, epsilon = 0.1, first_n = None, title = ""):
    for e in obj:
        y0, dy = e.equation
        if not first_n:
            first_n = 4*(tf-t0)
        tab = meth_epsilon_printer(y0, t0, tf, epsilon, dy, step, first_n)
        mp.plot(tab[:,0], tab[:,1], 'b')
    mp.xlabel("N(t)")
    mp.ylabel("P(t)")
    mp.title(title)
    mp.show()
    mp.clf()

def trajectory_grad(n0, m0, a, b, c, d, step, t0 = 0, tf = 5, epsilon = 0.1, first_n = None, radius = 0.5, by = 0.1, title = ""):
    if not first_n:
        first_n = 4*(tf-t0)
    for i in np.arange(-radius, radius, by):
        lv = lotka_volterra(n0 + i, m0 + i, a, b, c, d)
        y0, dy = lv.equation
        tab = meth_epsilon_printer(y0, t0, tf, epsilon, dy, step, first_n)
        mp.plot(tab[:,0], tab[:,1], 'r')
    mp.xlabel("N(t)")
    mp.ylabel("P(t)")
    mp.title(title)
    mp.show()
    mp.clf()

def period(obj, step, t0 = 0, tf = 5, epsilon = 0.1, first_n = None, precision = 0.01):
    y0, dy = obj.equation
    if not first_n:
        first_n = 4*(tf-t0)
    tab = meth_epsilon_printer(y0, t0, tf, epsilon, dy, step, first_n)
    x, y = tab[0,0], tab[0,1]
    for i in range(1, len(tab[:,0])):
        if (abs(tab[i,0]-x) < precision):
            if (abs(tab[i,1]-y) < precision):
                return i*(tf-t0)/float(len(tab[:,0]))
    return None

if __name__ == "__main__":
    # graph_time(malthus(50, 5), step_kutta, title = "Malthus (gamma > 0)")
    # graph_time(malthus(50, -1), step_kutta, title = "Malthus (gamma < 0)")
    graph_time(verhulst(1, 1, 5), step_kutta, title = "Verhulst de 1 a 5 individus")
    lv = lotka_volterra(1.5, 1.5, 1, 1, 1, 1)
    graph_time(lv, step_kutta, t0 = -5, tf = 20, first_n = 500, title = "Lotka-Volterra")
    trajectory([lv], step_kutta, t0 = 0, tf = 17, first_n = 500, title = "Trajectoire Lotka-Volterra")
    print("Periode : " + str(period(lv, step_kutta, t0 = 0, tf = 17, first_n = 500)))
    trajectory_grad(3, 3, 1, 1, 1, 1, step_kutta, t0 = -5, tf = 40, first_n = 500, title = "Trajectoires", radius = 2, by = 1)
