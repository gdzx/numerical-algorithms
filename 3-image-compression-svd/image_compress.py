#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as mp
import sys
import os.path
from bidiagonal_matrix import *
from qr_transform import *

def compress_matrix(A, k, graph_mode = False):
	""" Compression SVD d'une matrice quelconque A au rang k.
	"""
	if graph_mode:
		U, S, V = npl.svd(A, True)
		T = np.zeros((U.shape[1], V.shape[0]))
		T[:U.shape[1], :U.shape[1]] = np.diag(S)
		S = T
	else:
		Ql, BD, Qr = bidiagonal(A)
		U, S, V = toSVD(BD, 15, qr_decomposition_bidiag)
	for i in range(k + 1, len(S)):
		S[i,i] = 0
	if graph_mode:
		return np.dot(U, np.dot(S, V))
	return np.dot(np.dot(Ql,np.dot(U, np.dot(S, V))),Qr)

def compress(img, k):
	""" Compression d'une image 'img' au rang k.
	"""
	img = np.array(img)
	R = img[:, :, 0]
	G = img[:, :, 1]
	B = img[:, :, 2]
	img[:, :, 0] = compress_matrix(R, k).clip(0, 1)
	img[:, :, 1] = compress_matrix(G, k).clip(0, 1)
	img[:, :, 2] = compress_matrix(B, k).clip(0, 1)
	return img

if len(sys.argv) > 2:
	if not 'graphs' in sys.argv:
		try:
			img = mp.imread(sys.argv[1])
		except IOError as e:
			print(sys.argv[1] + ": " + e.strerror)
			sys.exit()
		res = compress(img, int(sys.argv[2]))
		mp.imshow(res)
		mp.imsave('tmp.png', res)
		print os.path.getsize('tmp.png')
		mp.show()
	else:
		try:
			img = mp.imread(sys.argv[1])
		except IOError as e:
			print(sys.argv[1] + ": " + e.strerror)
			sys.exit()
		s = os.path.getsize(sys.argv[1])
		it = []
		size_c = []
		size = []
		quality = []
		for i in range(0, len(img), 30):
			res = compress(img, i)
			mp.imsave('tmp.png', res)
			it.append(i)
			size_c.append(os.path.getsize('tmp.png'))
			size.append(s)
			quality.append(npl.norm(res - img))
		mp.xlabel("Rang de compression")
		mp.ylabel("Taille")
		mp.title("Variation de la taille de l'image en fonction du rang de compression")
		mp.plot(it, size_c)
		mp.plot(it, size)
		mp.legend(['Taille image compressee', 'Taille originale'])
		mp.show()

		mp.clf()

		mp.xlabel("Rang de compression")
		mp.ylabel("Distance entre l'image compressee et l'originale")
		mp.title("Qualite de l'image compressee")
		mp.plot(it, quality)
		mp.show()
else:
	print "Usage: ./" + os.path.basename(__file__) + " [image] [compression rank] graphs"