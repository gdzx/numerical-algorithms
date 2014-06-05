#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import unittest
import matplotlib.pyplot as mp
import sys
from householder_transforms import *

def qr_decomposition(A, mode = "", graph_mode = False):
	""" Décomposition QR d'une matrice
		Entrée:
			A matrice quelconque.
		Sortie:
			Tuple (Q, R) des matrices de la décomposition.
	"""
	if graph_mode:
		it = []
		decomp_quality = []
	A = np.matrix(A, dtype='f')
	N, M = A.shape
	Q = np.eye(N, N, dtype='f')
	R = np.matrix(A)
	for i in range(min(N, M)):
		X = R[i:, i]
		z = np.zeros([len(X), 1])
		if X[0, 0] > 0:
			z[0, 0] = npl.norm(X)
		else:
			z[0, 0] = -npl.norm(X)
		Q[:, i:] = householder_product_right(X, z, Q[:, i:])
		R[i:, :] = householder_product_left(X, z, R[i:, :])
		if graph_mode:
			it.append(i)
			Z = np.dot(Q, R)
			decomp_quality.append(npl.norm(Z - A))
	if graph_mode:
		return it, decomp_quality
	return Q, R

def qr_decomposition_bidiag(A, mode = "", graph_mode = False):
	""" Décomposition QR d'une matrice bidiagonale.
		Entrée:
			A matrice bidiagonale.
		Sortie:
			Tuple (Q, R) des matrices de la décomposition.
	"""
	A = np.matrix(A, dtype='f')
	N, M = A.shape
	Q = np.eye(N, dtype='f')
	R = A
	for i in range(min(N, M)-1):
		X = R[i:i+2, i]
		z = np.zeros([2, 1])
		if X[0, 0] > 0:
			z[0, 0] = npl.norm(X)
		else:
			z[0, 0] = -npl.norm(X)
		Q[:, i:i+2] = householder_product_right(X, z, Q[:, i:i+2])
		R[i:i+2, i:i+2] = householder_product_left(X, z, R[i:i+2, i:i+2])
	return Q, R

def toSVD(BD, iterations = 10, qr = npl.qr, graph_mode = False):
	""" Transformation SVD d'une matrice diagonale
		Entrée:
			BD matrice tridiagonale.
		Sortie:
			Un triplet correspondant aux matrices de la décomposition.
	"""
	if graph_mode:
		it = []
		decomp_quality = []
		convergence = []
	N, M = BD.shape
	U = np.eye(N)
	V = np.eye(M)
	S = np.matrix(BD)
	for i in range(iterations):
		Q1, R1 = qr(S.T, 'complete')
		Q2, R2 = qr(R1.T, 'complete')
		S = R2
		U = np.dot(U, Q2)
		V = np.dot(Q1.T, V)
		if graph_mode:
			it.append(i)
			Z = np.dot(np.dot(U, S), V)
			decomp_quality.append(npl.norm(Z - BD))
			convergence.append(npl.norm(np.diag(np.diag(S)) - S[:, :N]))
	if graph_mode:
		return it, decomp_quality, convergence
	for i in range(min(N, M)):
		if S[i,i] < 0:
			S[:, i] = -S[:, i]
			U[:, i] = -U[:, i]
	return U, S, V

if __name__ == '__main__':
	if "graphs" in sys.argv:
		# toSVD
		# BD = np.array([[1, 0, 0, 0], [2, -3, 0, 0], [0, -10, 5, 0]])
		BD = np.zeros([5, 7])
		for i in range(len(BD)):
			for j in range(len(BD[i])):
				if j <= i+1 and j >= i:
					BD[i, j] = npr.random()*100
		it, decomp_quality, convergence = toSVD(BD, 1000, qr_decomposition, True)
		mp.xlabel("Nombre d'iterations")
		mp.ylabel("Egalite B = U*S*V")
		mp.title("Verification de l'egalite BD = U*S*V")
		mp.plot(it, decomp_quality)
		mp.show()
		it, decomp_quality, convergence = toSVD(BD, 15, qr_decomposition_bidiag, True)
		mp.xlabel("Nombre d'iterations")
		mp.ylabel("Distance de S aux matrices diagonales")
		mp.title("Convergence de S vers une matrice diagonale")
		mp.plot(it[1:], convergence[1:])
		mp.show()

		mp.clf()

		# qr_decomp
		BD = np.zeros([10, 12])
		for i in range(len(BD)):
			for j in range(len(BD[i])):
				BD[i, j] = npr.random()*10
		it, decomp_quality = qr_decomposition(BD, 'complete', True)
		mp.xlabel("Nombre d'iterations")
		mp.ylabel("Egalite A = Q*R")
		mp.title("Verification de l'egalite A = Q*R")
		mp.plot(it, decomp_quality)
		mp.show()
	else:
		print "Usage: ./" + os.path.basename(__file__) + " graphs"