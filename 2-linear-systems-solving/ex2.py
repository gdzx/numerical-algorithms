#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg

# Question 3
def conjgrad (A, b, x = None, iterations = 10**6, epsilon = 10**(-10)):
    """
        Méthode du gradient conjugué.
        -----------------------------
        Entrée:
            A matrice symétrique définie positive.
            b vecteur colonne.
            (optional) x vecteur initial de la suite.
            (optional) iterations nombre d'itérations maximales pour apporcher la solution.
            (optional) epsilon précision minimale pour approcher la solution.
        Sortie:
            x solution approchée de Ax=b.
    """
    if not x:
        x = np.matrix(np.zeros([len(A),1]))

    r = b - A*x
    p = r
    rsold = (r.T * r)[0,0]
    rsnew = epsilon**2 + 1

    i = 1
    while i < iterations and np.sqrt(rsnew) > epsilon:
        Ap = A*p
        alpha= rsold/((p.T*Ap)[0,0])
        x = x + alpha*p
       #recuperer la valeur dans la matrice de taille 1.
        r = r - alpha*Ap
        rsnew = (r.T * r)[0,0]

        #print rsnew
        p = r + rsnew / (rsold*p)
        rsold = rsnew
        i+=1
    return x

# Question 4
def conjgrad_precond (A, b, M, x = None, iterations = 10**6, epsilon = 10**(-10)):
    """
        Méthode du gradient conjugué, avec préconditioneur.
        -----------------------------
        Entrée:
            A matrice symétrique définie positive.
            b vecteur colonne.
            M matrice préconditionneuse.
            (optional) x vecteur initial de la suite.
            (optional) iterations nombre d'itérations maximales pour apporcher la solution.
            (optional) epsilon précision minimale pour approcher la solution.
        Sortie:
            x solution approchée de Ax=b.
    """
    if not x:
        x = np.matrix(np.zeros([len(A),1]))
    xold = x
    rold = b - A*x
    zold = M.I*rold
    p = zold

    rnew = [epsilon**2 + 1]
    i = 1
    while i < iterations and numpy.linalg.norm(rnew) > epsilon:
        Ap = A*p
        alphaold = ((rold.T * zold)/(p.T*Ap))[0,0]
        xnew= xold + alphaold*p
        rnew = rold - alphaold*Ap
        znew = M.I*rnew
        betaold = (znew.T*rnew)[0,0] / (zold.T*rold)[0,0]
        p = znew + betaold*p

        rold = rnew
        zold = znew
        xold = xnew
        i+=1
    return xnew