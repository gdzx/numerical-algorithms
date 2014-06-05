#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
import newton_raphson as nr
import electrostatic_equilibrium as ee

U0 = np.zeros([1,1])
epsilon = pow(10, -30)
N = 1000

class TestSequenceFunctions(unittest.TestCase):
    def test_1(self):
        """ Roots of x^2 - 1 = 0 """
        def f(U):
            V = np.zeros([1,1])
            V[0,0] = pow(U[0,0], 2) - 1
            return V
        def j(U):
            V = np.zeros([1,1])
            V[0,0] = 2*U[0,0]
            return V
        U0[0,0] = 1025
        self.assertTrue(np.allclose(nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon), np.array([1])))
        U0[0,0] = -1025
        self.assertTrue(np.allclose(nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon), np.array([-1])))

    def test_2(self):
        """ Roots of x^2 = 0 """
        def f(U):
            V = np.zeros([1,1])
            V[0,0] = pow(U[0,0], 2)
            return V
        def j(U):
            V = np.zeros([1,1])
            V[0,0] = 2*U[0,0]
            return V
        nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon)
        U0[0,0] = 1025
        self.assertTrue(np.allclose(nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon), np.array([0])))
        U0[0,0] = -1025
        self.assertTrue(np.allclose(nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon), np.array([0])))

    def test_3(self):
        """ Roots of x^2 + 1 = 0 """
        N = 50
        def f(U):
            V = np.zeros([1,1])
            V[0,0] = pow(U[0,0], 2) + 1
            return V
        def j(U):
            V = np.zeros([1,1])
            V[0,0] = 2*U[0,0]
            return V
        nr.Newton_Raphson_Backtracking(f, j, U0, N, epsilon)
        self.assertTrue(1)

if __name__ == '__main__':
    unittest.main()