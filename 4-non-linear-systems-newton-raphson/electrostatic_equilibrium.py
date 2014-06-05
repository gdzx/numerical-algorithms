#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Libraries
import os
import sys
import numpy as np
import numpy.linalg as linalg
import random
import newton_raphson as nr
import numpy.polynomial.legendre as legendre
import matplotlib.pyplot as mp

### Problem
def random_charges(N):
    a = -1
    b = 1
    V = (b-a) * np.random.random_sample((N,)) + a
    np.append(V, a)
    np.append(V, b)
    return np.asmatrix(V).T

def energy_gradient(U):
    V = np.zeros([U.shape[0], U.shape[1]])
    for i in range(0, U.shape[0]):
        V[i,0] = (1.0)/(U[i,0] + 1) + (1.0)/(U[i,0] - 1)
        for j in range(0, U.shape[0]):
            if(j != i):
                V[i,0] = V[i,0] + (1.0)/(U[i,0] - U[j,0])
    return V

def energy_gradient_jacobian(U):
    V = np.zeros([U.shape[0], U.shape[0]])
    for i in range(0, U.shape[0]):
        for j in range(0, U.shape[0]):
            if(i != j):
                V[i,j] = (1.0)/(np.square(U[i,0] - U[j,0]))
            else:
                V[i,j] = -(1.0)/np.square(U[i,0] + 1) - (1.0)/np.square(U[i,0] - 1)
                for k in range(0, U.shape[0]):
                    if(k != i):
                        V[i,j] = V[i,j] - (1.0)/(np.square(U[i,0] - U[k,0]))
    return V

### Legendre polynomials
def legendre(x, d):
    N = np.zeros(d+1)
    N[d] = 1
    return np.polynomial.legendre.legval(x, np.polynomial.legendre.legder(N))

### Graphs
def graph1(epsilon, N, c):
    A = np.arange(-1, 1, 0.01)
    eig = []
    for i in range(1, c):
        U0 = random_charges(i)
        U = nr.Newton_Raphson_Backtracking(energy_gradient, energy_gradient_jacobian, U0, N, epsilon)
        eig.append([i, np.linalg.eigvals(energy_gradient_jacobian(U))])
        mp.plot([U[i,0] for i in range(U.shape[0])], [0 for i in range(U.shape[0])], 'o')
    for i in range(2, c+1):
        mp.plot(A, np.array(legendre(A, i)))
    mp.ylim(-5, 5)
    mp.axhline(0)
    mp.show()
    return eig

def graph2(eig):
    mp.xlim(0, len(eig) + 1)
    mp.xlabel("Number N of charges considered")
    mp.ylabel("Eigenvalue")
    for e in eig:
        mp.plot([e[0] for j in range(e[1].shape[0])], e[1], 'o')
    mp.show()

if __name__ == "__main__":
    if "graphs" in sys.argv:
        epsilon = pow(10, -10)
        N = 50
        eig = graph1(epsilon, N, 7)
        mp.clf()
        graph2(eig)
    else:
        print("Usage: ./" + os.path.basename(__file__) + " graphs")