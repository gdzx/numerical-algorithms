#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

def initprob(N):
    """
        Initialisation de la matrice liée au Laplacien.
        Entrée:
            N taille de la matrice (carrée).
    """

    dim = N * N
    indice = dim - N
    bloc1 = np.zeros([dim,dim]) # associé aux valeurs diagonales
    bloc2 = np.zeros([dim,dim]) # associé aux valeurs des deux blocs adjacents

    # remplissage
    for k in range(dim):
        bloc1[k,k] = 4
        if (k+1 < dim):
            bloc1[k+1, k] = -1
            bloc1[k, k+1] = -1
    for i in range(indice) :
        bloc2[N+i, i] = -1
        bloc2[i, N+i] = -1
    res = bloc1 + bloc2
    return res

def initb(N):
    """
        Initialisation du vecteur b au vecteur nul.
        Entrée:
            N taille de b.
    """
    dim = N*N
    res = np.zeros([dim,1])
    return res

def initradiateur(N, width = None):
    """
        Initialisation du vecteur b pour le cas du radiateur.
        Entrée:
            N taille de b.
    """

    if not width:
        width = int(np.ceil(np.sqrt(N)))

    dim = N*N
    res = np.zeros([dim,1])

    indice2 = (N-1)/2

    for k in range(indice2 - width / 2 + 1, indice2 + width / 2):
        for i in range(indice2 - width / 2 +1, indice2 + width / 2):
            res[k*N + i] = 100
    return res

def initmur(N):
    """
        Initialisation du vecteur b pour le cas du mur.
        Entrée:
            N taille de b.
    """
    dim = N*N
    res = np.zeros([dim,1])
    indicedebut = dim - (2 * N) + (N/6)
    indicefin = dim - N -(N/6)
    for i in range(indicedebut,indicefin):
            res[i] = 1
    return res

def convert(X):
    """
        Transformation d'un vecteur en matrice.
        Entrée:
            X vecteur.
        Sortie:
            Matrice correspondant au vecteur X selon la transformation proposée.
    """

    dim= int(np.sqrt(len(X)))
    res = np.zeros([dim,dim])
    pas = 1.0/(dim+1)
    #pas = 1
    for i in range(1,dim-1): # indices de manière a respecter les conditions de Dirichlet
        for j in range(1,dim-1): 
            res[i,j] = pas * X[(i * dim) + j]
    return res

def solutionq2(N):
    #solution de la question 2, on affiche l'apport de chaleur (vecteur b) et le resultat (vecteur x)
    res1 = npl.solve(initprob(N),initradiateur(N, 2)) #resolution du probleme
    entr = convert(initradiateur(N))  
    res2 = convert(res1) #transformation de x en matrice
    plt.imshow(entr, origin='lower', extent=(0,1,0,1)) #affichage entree
    plt.savefig("radiateur") 
    plt.imshow(res2, origin='lower',extent=(0,1,0,1)) #affichage solution
    plt.set_cmap('gist_heat');
    plt.title('Diffusion de la chaleur lorsque la source est un radiateur mis au centre')
    plt.show()


def solutionq3(N):
#solution de la question 3, on affiche l'apport de chaleur (vecteur b) et le resultat (vecteur x)        
    res1 = npl.solve(initprob(N),initmur(N)) #resolution du prob
    entr = convert(initmur(N)) 
    res2 = convert(res1)
    plt.imshow(entr, origin='lower', extent=(0,1,0,1)) #affichage entree
    plt.savefig("mur")
    plt.imshow(res2, origin='lower', extent=(0,1,0,1)) #affichage solution
    plt.set_cmap('gist_heat');
    plt.title('Diffusion de la chaleur lorsque la source est un radiateur mis au mur nord')
    plt.show()