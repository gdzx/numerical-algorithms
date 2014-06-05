#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import math
import time as t
import unittest
import matplotlib.pyplot as mp

# PARTIE 1 : décomposition de Cholesky
# Question 1
def cholesky(A):
    """
        Décomposition de Cholesky.
        --------------------------
        Entrée:
            A: matrice symétrique.
        Sortie:
            Matrice T approchée telle que T * transposée(T) = A
    """
    size = len(A)
    res = np.zeros([size,size])

    for j in range(size): # colonne par colonne
        for i in range(j, size):
            if (i == j):
                line_sum = 0
                for l in range(j):
                    line_sum += res[i,l]**2
                res[i,i] = np.sqrt(A[i,i] - line_sum)
            else:
                partial_sum = 0
                for k in range(i):
                        partial_sum += res[j,k] * res[i,k]
                res[i,j] = (A[i,j] - partial_sum) / res[j,j]
    return res

# Question 3
def sparseMatrix(n, p, max_value = 100):
    """
        Génération de matrices symétriques définies positives.
        ------------------------------------------------------
        Entrée:
            n taille de la matrice
            p nombre de termes extra-diagonaux non-nuls
            (optional) max_value valeurs maximale initiale
        Sortie:
            Une matrice n*n avec p termes extra-diagonaux non-nuls
    """
    iteration = (n**2 - n) / 2 - p

    # Random symmetric matrix
    A = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            A[i, j] = A[j, i] = (npr.rand()*max_value/2)*(-1)**(round(npr.rand())) + 1
    S = A + A.T

    k = l = 0 # random l and k indices where to put 0
    while (iteration > 0):
        lost = 0
        while (k == l) or S[k, l] == 0:
            l = k = np.rint((n-1) * (npr.rand()))
            while (k == l):
                l = np.rint((n-1) * (npr.rand()))
            lost += 1
            if lost > 5:
                i = 0
                while i < n and lost != 0:
                    j = 0
                    while j < n and lost != 0:
                        if S[i, j] != 0 and i != j:
                            lost = 0
                            k = i
                            l = j
                        j += 1
                    i += 1
        S[k, l] = S[l, k] = 0
        iteration -= 1

    # Transformation to a symetric definite positive matrix, as it is a diagonal dominant matrix
    A = np.floor(S + S.T)
    S = A + (np.floor(np.abs(np.amin(npl.eigvals(A))))+1)*np.eye(n)
    return S

# Question 4
def incompleteCholesky(A):
    """
        Décomposition de Cholesky incomplète.
        --------------------------
        Entrée:
            A: matrice symétrique.
        Sortie:
            Matrice T approchée telle que T * transposée(T) = A.
    """
    size = len(A)
    res = np.zeros([size,size])

    for j in range(size): # colonne par colonne
        for i in range(j, size):
            if A[i, j] != 0:
                if (i == j):
                    line_sum = 0
                    for l in range(j):
                        line_sum += res[i,l]**2
                    res[i,i] = np.sqrt(A[i,i] - line_sum)
                else:
                    partial_sum = 0
                    for k in range(i):
                            partial_sum += res[j,k] * res[i,k]
                    res[i,j] = (A[i,j] - partial_sum) / res[j,j]
    return res

# Questions 5/6
def initb(n, max_value = 10):
    """
        Initialise un vecteur de taille n avec des valeurs arbitraires.
    """
    res = np.zeros([n,1])
    for i in range(0,n):
        res[i] = npr.rand()*10
    return res

def count_zeros(A):
    """
        Nombres de zeros dans une matrice.
    """
    size = len(A)
    nb = 0
    for i in range(size):
        for j in range(size):
            if (A[i,j] == 0):
                nb += 1
    return nb

def compare_zeros(A, B):
    """
        Calcule la différence du nombre de zeros de deux matrices.
    """
    nb1 = count_zeros(A)
    nb2 = count_zeros(B)
    return abs(nb2 - nb1)

def gain_zeros(n, p):
    """
        Gain en nombre de zeros entre les deux méthodes.
    """
    A = sparseMatrix(n, p)
    T1 = cholesky(A)
    T2 = incompleteCholesky(A)
    res = compare_zeros(T1,T2)
    return res

def solution_dense(n,p,A,b):
    """
        Compare la résolution de Cholesky dense avec celle de linalg.solve.
    """
    T = cholesky(A)
    transposeT = np.transpose(T)
    invT = npl.inv(T)
    invtransposeT = npl.inv(T)
    invA = np.dot(invtransposeT, invT)
    x = np.dot(invA, b) # résolution
    xreel = npl.solve(A,b)
    erreur = xreel - x
    erreurabs = xreel - x
    for i in range(n):
        # calcul des erreurs relatives et absolues
        if (xreel[i] != 0):
            erreur[i] = abs((erreur[i]) / (xreel[i]))
            erreurabs[i] = abs(erreurabs[i])
    return erreurabs # erreur pour avoir l'erreur relative

def solution_incomplete(n,p,A,b):
    """
        Compare la résolution de Cholesky incomplet avec celle de linalg.solve.
    """
    T = incompleteCholesky(A)
    transposeT = np.transpose(T)
    invT = npl.inv(T)
    invtransposeT = npl.inv(T)
    invA = np.dot(invtransposeT, invT)
    x = np.dot(invA, b) # résolution
    xreel = npl.solve(A,b)
    erreurabs = xreel - x
    erreur = xreel - x
    for i in range(0,n):
        # calcul des erreurs relatives et absolues
        if (xreel[i] != 0):
            erreur[i] = abs((erreur[i]) / (xreel[i]))
            erreurabs[i] = abs(erreurabs[i])
    return erreurabs # erreur pour avoir l'erreur relative

def comparetempsresolution(n,p):
    """
        Compare les temps d'exécution des 2 méthodes de résolution.
    """
    A = sparseMatrix(n,p)
    b = initb(n)
    t1 = t.clock()
    dense = solution_dense(n,p,A,b)
    t2 = t.clock()
    incomplet = solution_incomplete(n,p,A,b)
    t3 = t.clock()
    res = abs((t2 - t1) - (t3 - t2))
    return res

def compare_results(n, p):
    """
        Retourne les résultat des 2 méthodes de résolution.
    """
    A = sparseMatrix(n,p)
    b = initb(n)
    dense = solution_dense(n,p,A,b)
    incomplet = solution_incomplete(n,p,A,b)
    return dense, incomplet

def condIncomplet(n, p, A):
    """
        Donne le conditionnement obtenu avec le préconditionneur (incomplet)
    """
    T = incompleteCholesky(A)
    transposeT = np.transpose(T)
    invT = npl.inv(T)
    invtransposeT = npl.inv(T)
    invA = np.dot(invtransposeT, invT)
    test1 = np.dot(invA,A)
    cond = npl.cond(test1)
    return cond

def conddense(n,p,A):
    """
        Donne le conditionnement obtenue avec le préconditionneur (dense)
    """
    T2 = cholesky(A)
    transposeT2 = np.transpose(T2)
    invT2 = npl.inv(T2)
    invtransposeT2 = npl.inv(T2)
    invA2 = np.dot(invtransposeT2, invT2)
    test2 = np.dot(invA2,A)
    cond =  npl.cond(test2)
    return cond

def preconditionneur(n, p):
    A = sparseMatrix(n,p)
    incomplet =  condIncomplet(n,p,A)
    dense = conddense(n,p,A)
    return npl.cond(A), dense, incomplet


def moyenne0(n,p,k):
    """
        Calcule le nombre de 0 gagnés avec k matrices de dimension n et ayant 2*p zeros
    """
    i = 1
    res = 0
    while (i <= k) :
        res += gain_zeros(n,p)
        i += 1
    res = res / (k * 1.0)
    return res

def graphmoyenne0(debut, n, p, k):
    """
        Affiche le graphe de moyenne0 debut = dimension de depart, n = dimension de fin, p nb de zero et k nb iteration a dim fixee pour la moyenne
    """
    tab = []
    t = np.arange(debut, n, 1)
    for i in range (0,len(t)):
        tab.append(moyenne0(t[i], p, k))
    mp.plot(t, tab, 'b', linewidth = 1.0)
    mp.xlabel('dimension de la matrice')
    mp.ylabel('moyenne du gain de 0 ')
    mp.show()

def moyenneTpsresolution(n,p,k):
    """
        Calcule le temps moyen supplementaire pour la resolution dense avec k matrices de dimension n et ayant 2*p zeros
    """
    i = 1
    res = 0
    while (i <= k) :
        res += comparetempsresolution(n,p)
        i += 1
    res = res / (k * 1.0)
    return res

def graphTpsresolution(debut, n, p, k):
    """
        Affiche le graphe de moyenneTpsresolution debut = dimension de depart, n = dimension de fin, p nb de zero et k nb iteration a dim fixee pour la moyenne
    """
    tab = []
    t = np.arange(debut, n, 1)
    for i in range (0,len(t)):
        tab.append(moyenneTpsresolution(t[i], p, k))
    mp.plot(t, tab, 'b', linewidth = 1.0)
    mp.xlabel('dimension de la matrice')
    mp.ylabel('temps moyen supplementaire pour la factorisation dense ')
    mp.show()

def moyenneprecond(n,p,k,A,m):
    """
        Calcule le preconditionnement moyen , m==0 pour incomplet, m == 1 pour dense
    """
    i = 1
    res = 0
    if (m == 1):
        while (i <= k) :
            res += conddense(n,p,A)
            i += 1
    elif (m == 0):
        while (i <= k) :
            res += condIncomplet(n,p,A)
            i += 1
    res = res / (k * 1.0)
    return res

def graphemoyenneprecond(debut, n, p, k):
    """
        Comparaison des conditionnements
    """
    tab1 = []
    tab2 = []
    tab3 = []
    t = np.arange(debut, n, 1)
    for i in range (0,len(t)):
        A = sparseMatrix(n,p)
        tab1.append(moyenneprecond(t[i], p, k, A, 0))
        tab2.append(moyenneprecond(t[i], p, k, A, 1))
        tab3.append(npl.cond(A))
    c1 = mp.plot(t, tab1, 'b', linewidth = 1.0,label='$Incomplet$')
    c2 = mp.plot(t, tab2,'g', linewidth = 1.0,label='$Dense$')
    c3 = mp.plot(t, tab3,'r', linewidth = 1.0,label='$cond(A)$')
    mp.xlabel('dimension de la matrice')
    mp.ylabel('conditionnement')
    mp.legend()
    mp.show()