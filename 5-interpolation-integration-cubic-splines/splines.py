#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages
import numpy as np
import matplotlib.pyplot as mp
import numpy.linalg as npl
import numpy.random as npr
import re
import matplotlib.pyplot as ma

epsilon = 0.0001

def load_foil(file):
  """ Airfoil: load profile of a wing
  Reads a file whose lines contain coordinates of points, separated by an empty line.
  Every line not containing a couple of floats is discarded.
  Returns a couple constitued of the list of points of the extrados and the intrados.
  """
  f = open(file, 'r')
  matchline = lambda line: re.match(r"\s*([\d\.-]+)\s*([\d\.-]+)", line)
  extra  = [];    intra  = []
  rextra = False; rintra = False
  for line in f:
    m = matchline(line)
    if (m != None) and not(rextra):
      rextra = True
    if (m != None) and rextra and not(rintra):
      extra.append(m.groups())
    if (m != None) and rextra and rintra:
      intra.append(m.groups())
    if (m == None) and rextra:
      rintra = True
  ex = np.array(list(map(lambda t: float(t[0]), extra))) # Python3: iterators to lists
  ey = np.array(list(map(lambda t: float(t[1]), extra)))
  ix = np.array(list(map(lambda t: float(t[0]), intra)))
  iy = np.array(list(map(lambda t: float(t[1]), intra)))
  return ex, ey, ix, iy

ex, ey, ix, iy = load_foil("e168.dat")

# Pour séparer extrados et intrados (bizarre d'ailleurs...)
ix = ex[len(ex)/2:]
iy = ey[len(ey)/2:]
ex = ex[:len(ex)/2+1]
ey = ey[:len(ey)/2+1]
# print(ex)
# print(ey)
# print(ix)
# print(iy)

def print_aile(ex, ey, ix, iy):
  mp.plot(ex, ey, 'ro')
  mp.plot(ix, iy, 'bo')
  mp.ylim([-0.3,0.3])

  # mp.show()
# print_aile(ex, ey, ix, iy)

def polynome_A(xj, xjj):
  """ Polynôme A associé au couple de point xj xj+1 (= xjj) """
  return np.poly1d([-1/(xjj - xj), xjj/(xjj - xj)])

def polynome_B(xj, xjj):
  """ Polynôme B associé au couple de point xj xj+1 (= xjj) """
  return np.poly1d([1/(xjj - xj), -xj/(xjj - xj)])

def C_aux(xj,xjj):
  """ Intermédiaire de calcul du polynôme C """
  return (pow(polynome_A(xj,xjj),3) - polynome_A(xj,xjj))/6

def polynome_C(xj, xjj):
  """ Polynome C associé au couple de point xj xj+1 (=xjj) """
  return (C_aux(xj,xjj) * pow((xjj - xj), 2))

def D_aux(xj,xjj):
  """ Intermédiaire de calcul du polynôme D """
  return (pow(polynome_B(xj,xjj),3) - polynome_B(xj,xjj))/6

def polynome_D(xj, xjj):
  """ Polynome D associé au couple de point xj xj+1 (=xjj) """
  return (D_aux(xj,xjj) * pow((xjj - xj), 2))

def system_aux(x,xj,xjj,y,yj,yjj):
  """ Code les équations (3.3.7) des "numericals recipes" sans le membre de droite """
  return [(xj-x)/6.0, (xjj - x) / 3.0, (xjj - xj) / 6.0]

def system_auxB(x,xj,xjj,y,yj,yjj):
  """ Code les équations (3.3.7) des "numericals recipes" (juste membre de droite) """
  return (yjj - yj)/(xjj - xj) - (yj - y)/ (xj - x)

def init_matrix(x,y):
  """ Initialise la matrice M pour résoudre le système (3.3.7) et obtenir y'' """
  n = len(x)
  res = np.zeros([n,n])
  res[0,0] = 1
  res[n-1,n-1] = 1
  for i in range(1,n-1):
    for j in range(0,2):
      res[i,i-1+j] = system_aux(x[i-1], x[i], x[i+1], y[i-1], y[i], y[i+1])[j]
  return res

x = np.array([10.0, 7.0, 3.0, 4.0, 18.0])
y = np.array([10.0, 8.0, 3.0, 5.0, 18.0])

def init_vector_B(x,y):
  """ Initialise le vecteur B pour le système M.X = B """
  dim = len(x)
  res = np.zeros([dim,1])
  res[0,0] = 0
  res[dim-1,0] = 0
  for i in range (1,dim-1):
    res[i,0] = system_auxB(x[i-1],x[i],x[i+1],y[i-1],y[i],y[i+1])
  return np.asmatrix(res)

# M = init_matrix(x, y)
# B = init_vector_B(x, y)

### la ligne suivante donne les y"
# print npl.solve(M,B)

def polynomials_by_seg(xj, xjj, yj, yjj, yj2, yjj2):
  """ Retourne le polynôme de degré 3 correspondant à l'intervalle [xj, xjj] """
  return ((polynome_A(xj, xjj)*yj) + (polynome_B(xj,xjj)*yjj) + (polynome_C(xj,xjj)*yj2) + (polynome_D(xj,xjj)*yjj2))

def sequence_polynomials(abs_list, ord_list):
  """ Retourne la liste des polynômes de degré 3 correspondant à une liste de points """
  ord_2_list = npl.solve(init_matrix(abs_list, ord_list), init_vector_B(abs_list, ord_list))
  poly_list = []
  for i in range(len(abs_list)-1):
    poly_list.append(polynomials_by_seg(abs_list[i], abs_list[i+1], ord_list[i], ord_list[i+1], ord_2_list[i,0], ord_2_list[i+1,0]))
  return poly_list

def derivate_sequence_polynomials(poly_list):
  """ Retourne la liste des dérivées des polynômes de degré 3 correspondant à une liste de points """
  deriv_list = []
  for i in range(len(poly_list)):
    deriv_list.append(derivate_poly_3(poly_list[i]))
  return deriv_list

def derivate_poly_3(poly):
  """ Fonction qui dérive tout polynôme de degré inférieur ou égal à trois """
  return np.poly1d([3 * poly[3] , 2 * poly[2] , poly[1] ])

def print_sequence_polynomials(poly_list, abs_list, ord_list, eps = epsilon, color = 'b'):
  """ Affiche une liste de polynômes correspondant à une liste de points """
  for i in range(len(poly_list)):
    print_polynomial (poly_list[i], min(abs_list[i], abs_list[i+1]), max(abs_list[i], abs_list[i+1]) ,eps, color)
  mp.ylim([-3*max(ey),3*max(ey)])


def print_polynomial(poly, mini, maxi, eps = epsilon, color = 'b'):
  """ Affiche un polynôme entre les bornes mini et maxi avec une précision souhaitée """
  x = np.arange(mini, maxi, eps)
  y = []
  for i in range(len(x)):
    y.append(poly(x[i]))
  mp.plot(x, y, color)
  # mp.show()

def test_C(xj,xjj):
  tmp = polynome_A(xj, xjj)
  tmp *=-1
  tmp += pow(polynome_A(xj, xjj), 3)
  tmp /= 6
  tmp *= pow((xjj-xj), 2)
  return polynome_C(xj, xjj) == tmp

# p = polynome_C(1, 3)

def test_splines():
    """ Test complet de splines.py """
    #mp.suptitle('Exemple de conditionnement suivant differentes matrices aleatoires')
    mp.subplot(121)

    #mp.ylabel('Cond(A)')
    #mp.xlabel('Pourcentage de valeurs non-nuls')
    print_sequence_polynomials(sequence_polynomials(ex,ey),ex,ey, epsilon,'r')
    print_sequence_polynomials(sequence_polynomials(ix,iy),ix,iy, epsilon,'b')
    print_aile(ex,ey,ix,iy)
    mp.title("Allure de l'aile")
    mp.subplot(122)

    print_sequence_polynomials(derivate_sequence_polynomials(sequence_polynomials(ex,ey)),ex,ey,epsilon, 'r')
    print_sequence_polynomials(derivate_sequence_polynomials(sequence_polynomials(ix,iy)),ix,iy,epsilon, 'b')
    mp.title("Allure des derives")
    #mp.xlabel('Pourcentage de valeurs non-nuls')
    #mp.ylabel('Cond(M-1 . A)')
    mp.show()
#test_splines()
