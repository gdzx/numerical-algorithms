#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as mp
import numpy.linalg as npl

def mathouseholder(x, y):
    """ Calcul de matrices de Householder.
        Entrée:
            (x, y) tuple de vecteurs de mêmes tailles.
        Sortie:
            H matrice de Householder de passage de x vers y.
    """
    n = x.size
    V = (x - y)
    if npl.norm(V) == 0:
        return np.eye(n)
    U = V / npl.norm(V)
    H = np.eye(n) - 2 * np.dot(U, U.T)
    return H

def _householder_product_left(X, Y, Z):
    """ Produit à gauche d'un vecteur par une matrice de Householder (naif).
        Entrée:
            (X, Y, Z) tuple de vecteurs de mêmes tailles.
        Sortie:
            Produit H*Z où H est la matrice de Householder de passage de X vers Y.
    """
    return np.dot(mathouseholder(X, Y), Z)

def householder_product_left(X, Y, M):
    """ Produit à gauche d'une matrice par une matrice de Householder.
        Entrée:
            X, Y, M respectivement deux vecteurs de mêmes tailles et une matrice.
        Sortie:
            Produit H*M où H est la matrice de Householder de passage de X vers Y.
    """
    V = X - Y

    if npl.norm(V) != 0:
        V = V / np.linalg.norm(V)

    return M - 2 * np.dot(V, np.dot(V.T, M))

def householder_product_right(X, Y, M):
    """ Produit à droite d'une matrice par une matrice de Householder.
        Entrée:
            X, Y, M respectivement deux vecteurs de mêmes tailles et une matrice.
        Sortie:
            Produit M*H où H est la matrice de Householder de passage de X vers Y.
    """
    V = X - Y

    if npl.norm(V) != 0:
        V = V / np.linalg.norm(V)

    return (M - 2 * np.dot(np.dot(M, V), V.T))