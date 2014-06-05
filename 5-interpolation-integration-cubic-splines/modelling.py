#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Packages
import numpy as np
import matplotlib.pyplot as mp
import numpy.linalg as npl
import numpy.random as npr
import matplotlib.pyplot as ma;

import splines as sp
import plane_curves_length as pcl

#########
#Données
#########

epsilon = 0.0001
ex, ey, ix, iy = sp.load_foil("e168.dat")
# Pour séparer extrados et intrados (bizarre d'ailleurs ...)
ix = ex[len(ex)/2:len(ex)]
iy = ey[len(ey)/2:len(ey)]
ex = ex[0:len(ex)/2+1]
ey = ey[0:len(ey)/2+1]

########################################################################
#Fonctions qui réalisent les calculs pour la partie supérieure de l'aile
########################################################################

def functions_list(f, step, hmax):
  """ Fonction qui calcule la liste des fonctions f(x,lambda) pour lambda entre 0 et 1 (partie supérieure de l'aile) """
  y = []
  for lambd in np.arange (0, 1 + step, step):
    y.append (curve_function(f, lambd, hmax))
  return y

def curve_function(f, lambd, hmax):
  """ Retourne le polynôme de f(x) appliqué à lambda (partie supérieure de l'aile)"""
  copy_f = []
  for i in range (len(f)):
    copy_f.append((np.poly1d((1-lambd) * f[i]) + lambd*3*hmax))
  return copy_f

def length_lambda_up(poly_list, abs_list):
  """ retourne la longueur d'une ligne d'air de l'aile"""
  length_result = 0
  derivate_list = sp.derivate_sequence_polynomials(poly_list)
  for i in range (len (abs_list)-1):
    length_result += pcl.length(derivate_list[i], pcl.uniform_subdivision,abs_list[i+1], abs_list[i], 2, pcl.simpson)
  return length_result

def image(abs_list, ex, poly_list):
  """ Calcule les images des polynomes représentant les trajectoires de l'air au dessus de l'aile"""
  cpt_x = 1
  ord_list = np.zeros(len(abs_list))
  i = 0
  while ((i < len (abs_list)) and (cpt_x < len(ex) -1)):
    while ((i < len (abs_list)) and (abs_list[i] >= ex[cpt_x])):
      ord_list[i] = poly_list[cpt_x-1](abs_list[i])
      i = i + 1
    cpt_x = cpt_x +1
  return ord_list
  

#########################################################################
# Fonctions qui réalisent les calculs pour la partie inférieure de l'aile
#########################################################################

def curve_function_down(f, lambd, hmax):
  """ Retourne le polynome de f(x) applique a lambda """
  copy_f = []
  for i in range (len(f)):
    copy_f.append((np.poly1d((1-lambd) * f[i]) - lambd*3*hmax))
  return copy_f

def functions_list_down(f, step, hmax):
  """ Fonction qui calcule la liste des fonctions f(x,lambda) pour lambda entre 0 et 1 (partie inférieure de l'aile)"""
  y = []
  for lambd in np.arange (0, 1 + step, step):
    y.append (curve_function_down(f, lambd, hmax))
  return y

def length_lambda_down(poly_list, abs_list):
  """ retourne la longueur d'une ligne d'air de l'aile (partie inférieure)"""
  length_result = 0
  derivate_list = sp.derivate_sequence_polynomials(poly_list)
  for i in range (len (abs_list)-1):
    length_result += pcl.length(derivate_list[i], pcl.uniform_subdivision,abs_list[i], abs_list[i+1], 2, pcl.simpson)
  return length_result

def image_down(abs_list, ix, poly_list):
  """ Calcule les images des polynomes représentant les trajectoires de l'air en dessous de l'aile"""
  cpt_x = 1
  ord_list = np.zeros(len(abs_list))
  i = 0
  while ((i < len (abs_list)) and (cpt_x < len(ix) -1)):
    while ((i < len (abs_list)) and (abs_list[i] < ix[cpt_x])):
      ord_list[i] = poly_list[cpt_x-1](abs_list[i])
      i = i + 1
    cpt_x = cpt_x +1
  return ord_list


###########################################################
#Fonctions qui permettent de tracer les différentes courbes
# ou cartes
###########################################################


def matrix_values (poly_list_up,poly_list_down,ex,ix):
  """ Retournes deux tableaux qui contiennent les longueurs des courbes de chaque côté de l'aile"""
  result_up = np.zeros(len(poly_list_up)-1)
  result_down = np.zeros(len(poly_list_down)-1)
  for i in range (len(poly_list_up)-1):
    result_up[i] = length_lambda_up (poly_list_up[i],ex)
    result_down[i] = length_lambda_down (poly_list_down[i],ix)
  return result_up, result_down

def matrix(ye, yi, result_up,result_down,ex,ey,ix,iy,step):
  """
  Fonction qui remplit la matrice représentant la carte de pression
  ye : fonctions représentant les f_lambda au dessus de l'aile
  yi : fonctions représentant les f_lambda en dessous de l'aile
  result_up : tableau qui contient la longueur des courbes au dessus de l'aile
  result_down : tableau qui contient la longueur des courbes en dessous de l'aile
  ex,ey : coordonnées des points au dessus de la courbe
  ix,iy : coordonnées des points en dessous de la courbe
  step : pas entre les différentes fonctions f_lambda
  """
  print "Calcul de la carte de pression en cours"
  solution = np.zeros([52,101]) #valeurs arbitraires à modifier si besoin
  position0 = 20                   #
  for k in range (0, len(ye)-1):
    abs_list = np.arange(1,0,-step)
    ord_list = image(abs_list, ex, ye[k])
    for j in range(0,len(abs_list)):
      solution[position0 + int (ord_list[j]*100),int(abs_list[j]*100)] = result_up[k]
  print "calcul de la carte de pression au dessus de l'aile terminé"
  for k in range (0, len(yi)-1):
    abs_list = np.arange(0,1,step)
    ord_list = image_down(abs_list, ix, yi[k])
    for j in range(0,len(abs_list)):
      solution[position0 -2 + int(ord_list[j]*100),int(abs_list[j]*100)] = result_down[k]
  i = 0
  print "calcul de la carte de pression en dessous de l'aile terminé"
  mp.imshow(solution, origin='lower', extent=(0,1,0,1), vmax=result_up[0], vmin=result_down[len(result_down)-1])
  mp.set_cmap('gist_heat');
  mp.title("Carte des pressions autour de l'aile")
  mp.show()



def affichage_lignes_air(ex, ey, ix, iy, step, hmax):
  """
  Fonction qui affiches les trajectoires de l'air autour de l'aile
  ex,ey : coordonnées des points au dessus de l'aile
  ix,iy : coordonnées des pointes en dessous de l'aile
  step : pas entre les différentes fonctions f_lambda
  hmax : hauteur max de l'aile
  """
  f1 = functions_list_down (sp.sequence_polynomials(ix,iy),step ,hmax)
  f = functions_list (sp.sequence_polynomials(ex,ey),step,hmax)
  print "Affichage des courbes en cours"
  for i in range (len(f)):
    sp.print_sequence_polynomials(f[i],ex,ey)
    sp.print_sequence_polynomials(f1[i],ix,iy)
    mp.title("Ecoulement laminaire de l'air autour de l'aile")
  mp.show()


def print_map (step,hmax):
  """
  Fonction qui affiche la carte de pression
  step : pas entre les différentes fonctions f_lambda
  hmax : hauteur max de l'aile
  """
  f1 = functions_list_down (sp.sequence_polynomials(ix,iy),step ,hmax)
  f = functions_list (sp.sequence_polynomials(ex,ey),step,hmax)
  l1,l2 = matrix_values(f,f1,ex,ix)
  m = matrix(f,f1,l1,l2,ex,ey,ix,iy,step)
  return m


affichage_lignes_air(ex,ey,ix,iy,0.05,max(ey))
print_map(0.005, max(ey))

