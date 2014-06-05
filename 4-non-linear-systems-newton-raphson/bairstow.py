#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages
import numpy as np
import matplotlib.pyplot as mp
import numpy.linalg as npl
import numpy.random as npr

# Comments
#
# In each function :
# P is a polynomial
# b, c are real numbers
# U is a vector representing the current position in Newton-Raphson method
# 

# Global Variables
epsilon = pow(10, -12)
N = 100

def bairstow_init(P, b, c):
    """
    defines the 2-dimension fonction f needed to use the Newton-Raphson method : f(b,c) = (R(b,c),S(b,c))
    """
    n = len(P)
    trinome = np.array([1, b, c])
    q1, r1 = np.polydiv(P,trinome)
    f2 = np.zeros([2,1])
    f2[0,0] = b
    f2[1,0] = c
    return f2,q1

def delta(U):
    """
    calculates the roots of a degre-2 polynomial : X2 - B X - C where U=[[B],[C]]
    returns the imaginary part and the real part of each root
    """
    delta = U[0,0] * U[0,0] + 4 * U[1,0]
    if (delta < 0):
        delta *= -1
        rproot1 =  U[0,0]/2 
        iproot1 =  np.sqrt(delta)  / 2 
        rproot2 =  U[0,0]/2 
        iproot2 = - np.sqrt(delta)  / 2 
    else:
        rproot1 = ( U[0,0] + np.sqrt(delta))  / 2
        rproot2 = ( U[0,0] - np.sqrt(delta)) / 2
        iproot1= 0
        iproot2 = 0
    return rproot1,iproot1, rproot2, iproot2

def delta2(a,b,c):
    """
    calculates the roots of a degre-2 polynomial : aX2 + b X + c
    returns the imaginary part and the real part of each root
    """
    delta = b * b - 4 * c *a
    if (delta < 0):
        delta *= -1
        rproot1 =  (-b)/(2*a)
        iproot1 =  np.sqrt(delta)  / (2*a)
        rproot2 =  (-b)/(2*a)
        iproot2 =  - np.sqrt(delta)  / (2*a)
    else:
        rproot1 = (- b + np.sqrt(delta))  / (2 * a)
        rproot2 = (- b - np.sqrt(delta)) / (2 *a)
        iproot1 = 0
        iproot2 = 0
    return rproot1,iproot1, rproot2, iproot2

def erreurR(U,V):
    """
    computes the relative X-coordinate error related to a step in Newton-Raphson method
    - V, U : vectors
    """
    return abs(V[0,0]/(U[0,0]+V[0,0]))

def erreurS(U,V):
    """
    computes the relative Y-coordinate error related to a step in Newton-Raphson method
    - V, U : vectors
    """
    return abs(V[1,0]/(U[1,0]+V[1,0]))

def init_bk(P,b,c):
    """
    initializes the coefficient of Q(x) after the division of P
    Needed in Newton-Raphson method
    """
    n = len(P)
    i = n-3
    solution = np.zeros([1,n])
    solution[0,n-1] = P[0]
    solution[0,n-2] = P[1] + b * solution[0,n-1]
    while i > -1:
        solution[0,i] = P[n-i-1] + b * solution[0,i+1] + c * solution[0,i+2]
        i -= 1
    return solution

def init_ck(P,Tbk,b,c):
    """
    initializes the coefficients which enables to calculate partial derivatives in Newton-Raphson method
    """
    n = len(P)
    i = n-3
    solution = np.zeros([1,n])
    solution[0,n-1] = Tbk[0,n-1]
    solution[0,n-2] = Tbk[0,n-2] + b * solution[0,n-1]
    while i > 0:
        solution[0,i] = Tbk[0,i] + b * solution[0,i+1] + c * solution[0,i+2]
        i-=1
    return solution
    
def init_deriv(P,Tbk,Tck):
    """
    Calculates the partial derivatives of the function f used in Newton-Raphson method
    """
    partialRC = Tck[0,3]
    partialSC = Tck[0,2]
    partialRB = Tck[0,2]
    partialSB = Tck[0,1]
    return partialRC, partialSC, partialRB, partialSB

def next_position3(P,U):
    """
    computes the direction towards a root by using the jacobian matrix det
    Needed in Newton-Raphson method
    """
    tbk = init_bk(P,U[0,0],U[1,0])
    tck = init_ck(P,tbk,U[0,0],U[1,0])
    RC,SC,RB,SB = init_deriv(P,tbk,tck)
    Jacobian_det = (RB * SC) - (RC * SB)
    solution = np.zeros([2,1])
    solution[0, 0] =  -(1.0 / Jacobian_det) * (tbk[0,1] * SC - tbk[0,0] * RC) 
    solution[1, 0] =  -(1.0 / Jacobian_det) * (tbk[0,0] * RB - tbk[0,1] * SB) 
    return solution

def Newton2(P, N, U, epsilon):
    """
    Uses Newton-Raphson method to find a quadratic factor of P
    P : polynomial; N maximum number of steps;U : vector (initial position); epsilon : precision
    """
    j = 0
    U = np.zeros([2,1])
    V = next_position3(P,U) 
    while (j < N and erreurR(U,V) > epsilon and erreurS(U,V) > epsilon):
        U = U + V
        V = next_position3(P,U)
        j += 1
    return U

def Bairstow(P,b,c,N,epsilon):
    """
    finds the roots of P by using bairstow method and keeps them in an array
    P polynomial; b,c real; N : maximum number of steps; epsilon : precision
    """
    n = len(P) - 1
    roots = np.zeros([1,2*n])
    i = 0
    while (len(P) > 3):
        U,Q = bairstow_init(P,-b,-c)
        V = Newton2(P,N,U,epsilon)
        roots[0,i],roots[0,i+1],roots[0,i+2],roots[0,i+3] = delta(V)
        i+=4
        P = bairstow_init(P,-V[0,0],-V[1,0])[1]
    if (len(P) == 3):
        roots[0,i],roots[0,i+1],roots[0,i+2],roots[0,i+3] = delta2(P[0],P[1],P[2])
    elif (len(P) == 2):
        roots[0,i] = -P[1] / P[0]
    else:
        roots[0,i] = 0
    return roots

# Tests

#polynomials with real roots
p = np.array([1,-11,32,-22]) 
p2 = np.array([1,-5,10,-10,4])

def polynomial_evaluation(P,x,y):
    """
    returns the evaluation of P at x (+ y i if complex)
    """
    value = 0
    n = len(p)
    deg = n
    i = 0
    while -1 != deg :
        value += P[i] * pow(x + y * 1j,deg)
        deg -= 1
        i+=1
    return value


def verify_roots(P,roots):
    """
    tests all the roots of a polynomial P
    """
    test = True
    i = 0
    while (i <= len(roots) and test):
        tmp = polynomial_evaluation(P,roots[0,i],roots[0,i+1])
        print tmp
        if (abs (tmp) > 0.01):
            test = False
        else:
            i+=2
    return test

def init_polynomial(n,m):
    """
    creates a polynomial of degre n
    """
    p = np.zeros([1,n+1])
    for i in range(0,n+1):
        p[0,i] = np.rint((m) * (npr.rand())) + 1
    return p[0,:]

def test_bairstow(n,m,b,c):
    """
    verifies Bairstow method.
    """
    p = init_polynomial(n,m)
    #print p
    roots = Bairstow(p, b, c, N, epsilon)
    #print roots
    return verify_roots(p,roots)