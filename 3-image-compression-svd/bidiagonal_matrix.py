#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as mp
from householder_transforms import *

def bidiagonal(A):
    """ Réduction de matrices sous forme bidiagonale.
        Entrée:
            A matrice quelconque.
        Sortie:
            Tuple (Ql, BD, Qr) :
                - Ql matrice de changement de base à gauche,
                - BD matrice bidiagonale,
                - Qr matrice de changement de base à droite.
    """
    A = np.matrix(A, dtype='f')
    n, m = A.shape
    Qleft = np.eye(n, dtype='f')
    Qright = np.eye(m, dtype='f')
    BD = np.matrix(A)
    for i in range(n):
        x = BD[i:, i]
        y = np.zeros([n-i, 1])
        if x[0, 0] > 0:
            y[0, 0] = npl.norm(x)
        else:
            y[0, 0] = -npl.norm(x)
        Qleft[i:, i:] = householder_product_right(x, y, Qleft[i:, i:])
        BD[i:, i:] = householder_product_left(x, y, BD[i:, i:])
        if i < n-1:
            x = BD[i, i+1:].T
            y = np.zeros([m-(i+1), 1])
            if x[0, 0] > 0:
                y[0, 0] = npl.norm(x)
            else:
                y[0, 0] = -npl.norm(x)
            Qright[i+1:, i:] = householder_product_left(x, y, Qright[i+1:, i:])
            BD[i:, i+1:] = householder_product_right(x, y, BD[i:, i+1:])
    return Qleft, BD, Qright