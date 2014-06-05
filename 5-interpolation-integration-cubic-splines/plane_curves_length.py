#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Packages
import numpy as np
import unittest
import numpy.polynomial.polynomial
import matplotlib.pyplot as plt

### Subdivision
def uniform_subdivision(n, a, b):
	""" Return the [a,b] interval subdivision in n equal segemnts """
	p = (b-a)*1./n
	return np.arange(a, b + p, p)

### Fonctions
def f(x):
	return 1

### Integration method
def rectangle(f, a, b, i):
	"Méthode générique des rectangles"
	return (b-a)*f(i)

def rectangle_inf(f, a, b, n_eval = False):
	"Méthode des rectangles inférieurs"
	if n_eval:
		return 1
	return rectangle(f, a, b, a)

def rectangle_sup(f, a, b, n_eval = False):
	"Méthode des rectangles supérieurs"
	if n_eval:
		return 1
	return rectangle(f, a, b, b)

def middle(f, a, b, n_eval = False):
	"Méthode du point milieu"
	if n_eval:
		return 1
	m = (a+b)/2.
	return rectangle(f, a, b, m)

def trapeze(f, a, b, n_eval = False):
	"Méthode des trapèzes"
	if n_eval:
		return 2
	return (b-a)*(f(a)+f(b))/2.

def simpson(f, a, b, n_eval = False):
	"Méthode de Simpson"
	if n_eval:
		return 3
	m = (a+b)/2.
	return (b-a)/6. * (f(a) + 4*f(m) + f(b))

### Integral
def integral_rect(f, subdivision = uniform_subdivision, t_0 = 0, T = 1, n = 2, method = simpson, n_eval = False):
	"Calcul d'intégral modulaire"
	sub = subdivision(n, t_0, T)
	I = 0
	it = method(f, t_0, T, True)
	N = (len(sub)-1)*it
	for i in range(len(sub)-1):
		a = sub[i]
		b = sub[i+1]
		I += method(f, a, b)
	if n_eval:
		return I, N
	return I
#print(integral_rect(f, uniform_subdivision, 0, 10, 46, rectangle_inf))

### calcul de somme
def sum(f, a, b, n):
    s = 0
    h = (b - a)/(n * 2.)
    k = 0
    while k < n :
        s = s + f (a + h + 2 * k * h)
        k = k + 1
    return s

### Integral optimised
def integral_optimised(f, a, b, method_I, epsilon):
	"Calcul d'intégrale optimisé"
	n = 2
	I_n = integral_rect(f, t_0 = a, T = b, n = n, method = method_I)
	n *= 2
	I_2n = integral_rect(f, t_0 = a, T = b, n = n, method = method_I)
	while np.abs (I_2n - I_n) > epsilon:
		I_n = I_2n
		I_2n = (1./2)*I_n + ((b - a)/(2*n)) * sum(f, a, b, n)
		n = 2 * n
	return I_2n

def integral_optimised_simpson (f, a, b, epsilon = 10 ** (-6)):
	"Calcul d'intégrale optimisé (Simpson)"
	F = np.array([f(a),f((a+b)/2.),f(b)])
	I_2n = (b - a)*(F[0]+4*F[1]+F[2])/6.
	h = np.abs(b-a)
	diff = 1
	while (diff > epsilon):
		n = np.size(F)
		I_n = I_2n
		h = h/2.
		tmp = np.zeros([2*n-1])
		for i in range(n-1):
			#on garde les termes deja calcule de f
			tmp[2 * i] = F[i]
			tmp[1 + 2 * i] = f(a + h * (i + 0.5))
		tmp[2*n-2] = F[n-1]
		F = tmp
		I_2n = 0
		for i in range(n-1):
			I_2n = I_2n + (F[i * 2]+4*F[i * 2 + 1] + F[i * 2 + 2])
		I_2n = I_2n * h /6.
		diff = np.abs(I_n - I_2n)
	return I_2n

### Length
def length(f_prime, subdivision = uniform_subdivision, t_0 = 0, T = 1, n = 2, method = simpson):
	F = lambda x: np.sqrt(1 + (f_prime(x))**2)
	return integral_rect(F, subdivision,t_0, T, n, method)
#print(length(f, T = 1, n = 2)) -> sqrt(2)

def graph1():
	methods = [rectangle_inf, middle, trapeze, simpson]
	methods_name = ['Rectangle', 'Point millieu', 'Trapèze', 'Simpson']
	res = []
	for m in methods:
		D = np.array([])
		E = np.array([])
		r = 20
		for i in range(50, 300):
			I, N = integral_rect(lambda x: np.exp(x), uniform_subdivision, 0, r, i, m, True)
			D = np.append(D, I-np.exp(r)+1)
			E = np.append(E, i)
		res.append([D, E])

	fig, ax = plt.subplots()
	for e in res:
		ax.plot(np.log(e[1]), np.log(np.abs(e[0])), 'o')
	plt.legend(methods_name)
	plt.xlabel("log(n)")
	plt.ylabel("log(erreur)")
	plt.title("Comparaison des différentes méthodes d'intégration")
	plt.show()

def graph2():
	methods = [rectangle_inf, middle, trapeze, simpson]
	methods_name = ['Rectangle', 'Point millieu', 'Trapèze', 'Simpson']
	res = []
	for m in methods:
		D = np.array([])
		E = np.array([])
		r = 20
		for i in range(50, 300):
			I, N = integral_rect(lambda x: np.exp(x), uniform_subdivision, 0, r, i, m, True)
			D = np.append(D, I-np.exp(r)+1)
			E = np.append(E, N)
		res.append([D, E])

	fig, ax = plt.subplots()
	for e in res:
		ax.plot(np.log(e[1]), np.log(np.abs(e[0])), 'o')
	plt.legend(methods_name)
	plt.xlabel("log(nombre d'évaluation de f)")
	plt.ylabel("log(erreur)")
	plt.title("Comparaison des différentes méthodes d'intégration")
	plt.show()

if __name__ == '__main__':
	graph1()
	graph2()
