#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import unittest
import matplotlib.pyplot as mp
import sys
from householder_transforms import *
from qr_transform import *
from bidiagonal_matrix import *

def isBidiagonal(M, epsilon = 10**-15):
	M = np.matrix(M)
	for i in range(M.shape[0]):
		for j in range(M.shape[1]):
			if (not (i-1 <= j <= i+1)) and abs(M[i, j]) > epsilon:
				return False
			if i == M.shape[0]-1:
				return True
	return True

def randomMatrix(n, m):
	M = np.zeros([n, m])
	for i in range(n):
		for j in range(m):
			M[i, j] = npr.random()*100*(-1)**(np.rint(npr.random()))
	return M

def randomBidiagonalMatrix(n, m):
	M = np.zeros([n, m])
	for i in range(n):
		for j in range(m):
			if i-1 <= j <= i:
				M[i, j] = npr.random()*100*(-1)**(np.rint(npr.random()))
	return M

class TestHouseholderTransforms(unittest.TestCase):
    def setUp(self):
        self.x = np.matrix([3, 4, 0]).T
        self.y = np.matrix([0, 0, 5]).T
        self.H = mathouseholder(self.x, self.y)

    def test_mathouseholder(self):
        self.assertTrue(np.array_equal(np.dot(self.H, self.x), self.y))

    def test_householder_product_left(self):
    	self.assertTrue(np.array_equal(householder_product_left(self.x, self.y, self.x), self.y))

    def test_householder_product_right(self):
        self.assertTrue(np.allclose(householder_product_right(self.x, self.y, self.y.T).T, self.x, 10**-10, 10**-10))

class TestBidiagonalMatrix(unittest.TestCase):
	@unittest.expectedFailure
	def test_bidiagonal(self):
		BD = np.eye(30, 35, 0) + np.eye(30, 35, 1)
		self.assertTrue(isBidiagonal(BD))
		_BD = bidiagonal(BD)
		for i in range(2):
			self.assertTrue(isBidiagonal(_BD[i]))

		M = randomMatrix(30, 35)
		_M = bidiagonal(M)
		self.assertTrue(isBidiagonal(_M[1], 10**-4))
		self.assertTrue(np.allclose(np.dot(np.dot(_M[0], _M[1]), _M[2]), M, 1, 10))

class TestQRTransform(unittest.TestCase):
	def test_qr_decomposition(self):
		M = randomMatrix(30, 35)
		_M = qr_decomposition(M)
		self.assertTrue(np.allclose(np.dot(_M[0], _M[1]), M, 10**-4, 10**-5))

	def test_to_SVD(self):
		M = randomBidiagonalMatrix(30, 35)
		_M = toSVD(M, 5000)
		self.assertTrue(np.allclose(np.dot(np.dot(_M[0], _M[1]), _M[2]), M, 10**-4, 10**-8))

if __name__ == '__main__':
    unittest.main()