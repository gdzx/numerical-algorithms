#!/usr/bin/env python
# -*- coding: utf-8 -*-

### Libraries
import sys
import os
import numpy as np
import newton_raphson as nr

# Plot tools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

### Forces builders
def build_elastical_force(intensity):
	""" Build and return the elastical force and its Jacobian """
	def fe(p):
		l = np.sqrt(p[0,0]**2+p[1,0]**2)
		if l == 0:
			return np.matrix([[0], [0]])
		return intensity*p/l
	def hfe(p):
		l = np.sqrt(p[0,0]**2+p[1,0]**2)
		if l == 0:
			return np.matrix([[0, 0], [0, 0]])
		e = intensity/l
		return np.matrix([[e, 0], [0, e]])
	return fe, hfe

def build_centrifugal_force(intensity, origin):
	""" Build and return the centrifugal force and its Jacobian """
	def fc(p):
		d = p-origin
		return intensity*d
	def hfc(p):
		return np.matrix([[intensity, 0], [0, intensity]])
	return fc, hfc

def build_gravitational_force(intensity, origin):
	""" Build and return the gravitational force and its Jacobian """
	def fg(p):
		d = p-origin
		l = ((p[0,0]-origin[0,0])**2 + (p[1,0]-origin[1,0])**2)**(3./2)
		if l == 0:
			return np.matrix([[float("inf")], [float("inf")]])
		return -intensity*d/l
	def hfg(p):
		d = p-origin
		l = ((origin[0,0] - p[0,0])**2 + (origin[1,0] - p[1,0])**2)**(5./2)
		if l == 0:
			return np.matrix([[0, 0], [0, 0]])
		return np.matrix([
			[(intensity*(2*origin[0,0]**2 - origin[1,0]**2 - 4*origin[0,0]*p[0,0] + 2*p[0,0]**2 + 2*origin[1,0]*p[1,0] - p[1,0]**2))/l,
			(3*intensity*(origin[0,0] - p[0,0])*(origin[1,0] - p[1,0]))/l],
			[(3*intensity*(origin[0,0] - p[0,0])*(origin[1,0] - p[1,0]))/l,
			-((intensity*(origin[0,0]**2 - 2*origin[1,0]**2 - 2*origin[0,0]*p[0,0] + p[0,0]**2 + 4*origin[1,0]*p[1,0] - 2*p[1,0]**2))/l)]])
	return fg, hfg

### Situation
class situation():
	def __init__(self, origin1, origin2, intensity1, intensity2, intensity3):
		self.origin1 = origin1
		self.origin2 = origin2
		self.intensity1 = intensity1
		self.intensity2 = intensity2
		self.intensity3 = intensity3
		self.origin3 = self.intensity2 / (self.intensity1 + self.intensity2) * self.origin2 - self.origin1
		self.fg1, self.hfg1 = build_gravitational_force(self.intensity1, self.origin1)
		self.fg2, self.hfg2 = build_gravitational_force(self.intensity2, self.origin2)
		self.fc, self.hfc = build_centrifugal_force(self.intensity3, self.origin3)

	def F(self, p):
		return self.fg1(p) + self.fg2(p) + self.fc(p)

	def H(self, p):
		return self.hfg1(p) + self.hfg2(p) + self.hfc(p)

	def fun(self, x, y, lim = float('inf'), log = False):
		""" Return the log of norm of F at (x, y), or lim if exceded """
		p = np.matrix([[x], [y]], dtype='f')
		N = np.linalg.norm(self.F(p))
		if log:
			return min(lim, np.log(N))
		return min(lim, N)

### Graphs
def graph1():
	### Case setup
	origin1 = np.matrix([[0.], [0]])
	origin2 = np.matrix([[1.], [0]])
	S = situation(origin1, origin2, 1, 0.01, 1)

	### Ploting
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X = np.arange(-1.5, 2, 0.05)
	Y = np.arange(-1.5, 1.5, 0.05)
	X, Y = np.meshgrid(X, Y)
	zs = np.array([S.fun(x, y, lim = 10) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)
	surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	ax.set_zlim(0, 10)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=20)
	plt.show()

def graph2():
	### Case setup
	origin1 = np.matrix([[0.], [0]])
	origin2 = np.matrix([[1.], [0]])
	S = situation(origin1, origin2, 1, 0.5, 1)

	# Ploting
	fig, ax = plt.subplots()
	X = np.arange(-1.5, 2, 0.01)
	Y = np.arange(-1.5, 1.5, 0.01)
	X, Y = np.meshgrid(X, Y)
	zs = np.array([S.fun(x, y, lim = 5, log = True) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)
	surf = ax.pcolor(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	CS = plt.contour(X, Y, Z)
	fig.colorbar(surf, shrink=0.5, aspect=20)
	plt.show()

def graph3():
	### Case setup
	origin1 = np.matrix([[0.], [0]])
	origin2 = np.matrix([[1.], [0]])
	S = situation(origin1, origin2, 1, 0.01, 1)

	# Ploting
	fig, ax = plt.subplots()
	X = np.arange(-1.5, 2, 0.01)
	Y = np.arange(-1.5, 1.5, 0.01)
	X, Y = np.meshgrid(X, Y)
	zs = np.array([S.fun(x, y, lim = 5, log = True) for x, y in zip(np.ravel(X), np.ravel(Y))])
	Z = zs.reshape(X.shape)
	surf = ax.pcolor(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
	CS = plt.contour(X, Y, Z)
	C = []
	for i in np.arange(-1.5, 2, 0.2):
		for j in np.arange(-1.5, 2, 0.2):
			N = nr.Newton_Raphson(S.F, S.H, np.matrix([[i], [j]]), 1000, 10**-10)
			if not C:
				C.append(N)
			f = False
			for item in C:
				if np.linalg.norm(item-N) < 10**-5:
					f = True
					break
				if not f:
					C.append(N)
	plt.plot([i[0,0] for i in C], [i[1,0] for i in C], 'bo')
	plt.plot([origin1[0,0]], [origin1[1,0]], 'ro')
	plt.plot([origin2[0,0]], [origin2[1,0]], 'ro')
	plt.show()

if __name__ == "__main__":
	if "graphs" in sys.argv:
		print("<!> It can takes up to one minute to plot a graph.")
		F = [graph1, graph2, graph3]
		for function in F:
			function()
	else:
		print("Usage: ./" + os.path.basename(__file__) + " graphs")