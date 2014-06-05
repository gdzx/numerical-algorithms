#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Libraries
import os
import sys
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as mp

### Newton-Raphson
def Newton_Raphson(F, J, U0, N, epsilon):
    j = 0
    U = U0
    V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
    while (j < N and linalg.norm(V) > epsilon):
        U = U + V
        V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
        j += 1
    return U

def Newton_Raphson_Backtracking(F, J, U0, N, epsilon):
    j = 0
    U = U0
    V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
    while (j < N and linalg.norm(F(U)) > epsilon):
        while linalg.norm(F(U + V)) > linalg.norm(F(U)): # Backtracking
            V = V * 2 / 3
        U = U + V
        V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
        j += 1
    return U

### Graphs
def speed_Newton_Raphson_Backtracking(F, J, U0, N, epsilon):
    j = 0
    U = U0
    V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
    speed = []
    iterations = []
    while (j < N and linalg.norm(F(U)) > epsilon):
        while linalg.norm(F(U + V)) > linalg.norm(F(U)):  # Backtracking
            V = V * 2 / 3
        U = U + V
        c = linalg.norm(V)
        speed = np.append(speed,c)
        iterations = np.append(iterations, j+1)
        V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
        j += 1
    mp.plot(np.log(iterations),np.log(speed),'o')
    j = 0
    U = U0
    V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
    speed1 = []
    iterations1 = []
    while (j < N and linalg.norm(F(U)) > epsilon):
        U = U + V
        c = linalg.norm(V)
        speed1 = np.append(speed1,c)
        iterations1 = np.append(iterations1, j+1)
        V = np.asmatrix(linalg.lstsq(J(U), -F(U))[0])
        j += 1
    mp.plot(np.log(iterations1),np.log(speed1))
    mp.xlabel("log(number of iterations)")
    mp.ylabel("log(speed of convergence)")
    mp.legend(["backtracking", "nobacktracking"])
    mp.grid(True)
    mp.show()

if __name__ == "__main__":
    if "graphs" in sys.argv:
        # Setup
        U0 = np.zeros([1,1])
        U0[0,0] = 1025
        epsilon = pow(10, -10)
        N = 10000
        def f(U):
            V = np.zeros([1,1])
            V[0,0] = pow(U[0,0],3) - 1
            return V
        def j(U):
            V = np.zeros([1,1])
            V[0,0] = 3*pow(U[0,0],2)
            return V

        # Ploting
        speed_Newton_Raphson_Backtracking(f,j,U0,N,epsilon)
        mp.clf()
        U0[0,0] = 0.25
        speed_Newton_Raphson_Backtracking(f,j,U0,N,epsilon)
    else:
        print("Usage: ./" + os.path.basename(__file__) + " graphs")