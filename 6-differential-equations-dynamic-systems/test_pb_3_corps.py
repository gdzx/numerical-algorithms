# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as mp
import resolution as p1
import pb_3_corps as pb



#########################################################################
# problème de trois corps
# paramètres : a0, c0 vecteurs définissant la position et la vitesse
#              initiales de a et de c
#              la masse de a et la masse de b
#              le pas :h, et le nombre d'itérations N
#

print
print
print "Testing three body problem"        






print "In ordinary base"

##########################################################################

t = np.arange(0,(16.*np.pi*24*3600),10000)
x = np.zeros(len(t))
y = np.zeros(len(t))
for i in range(0,t.size):
    x[i] = (lambda t: np.cos((1/(2.*np.pi*24*3600)) * t))(t[i])
    y[i] = (lambda t: np.sin((1/(2.*np.pi*24*3600)) * t))(t[i])
a0 = np.array([0,0,0,0])
b0 = np.array([lambda t: np.cos((1/(2.*np.pi*24*3600)) * t),lambda t: np.sin((1/(2.*np.pi*24*3600)) * t)])
c0 = np.array([-0.5,0,0,-1])
masse_a = 1
masse_b = 0.01
N = 50000
h = 0.001
tab = pb.three_body_problem(a0,c0,b0,masse_a,masse_b,N,h)
mp.plot(tab[:,0],tab[:,1])
mp.plot(c0[0],c0[1],'go')
mp.plot(x,y)
mp.scatter(a0[0],a0[1],c="y",s=100) # corps A
mp.show()


###########################################################################

print "Base change"
a0 = np.array([0,0,0,0])
b0 = np.array([lambda t: np.cos(t),lambda t: np.sin(t)])
c0 = np.array([-0.75,0.1,1,-1])
masse_a = 1
masse_b = 0.01
N = 12000
h = 0.001
tab0 = pb.three_body_problem_base_change(a0,c0,b0,masse_a,masse_b,N,h)
mp.plot(np.asarray(tab0[0]),np.asarray(tab0[1]))
mp.plot(c0[0],c0[1],'go')

a1 = np.array([0,0,0,0])
b1 = np.array([lambda t: np.cos(t),lambda t: np.sin(t)])
c1 = np.array([-0.7,0,1,-1])
masse_a1 = 1
masse_b1 = 0.01
N1 = 12000
h1 = 0.001
tab1 = pb.three_body_problem_base_change(a1,c1,b1,masse_a1,masse_b1,N1,h1)
mp.plot(np.asarray(tab1[0]),np.asarray(tab1[1]))
mp.plot(c1[0],c1[1],'go')
mp.scatter(a0[0],a0[1],c="y",s=100) # corps A
mp.scatter (1,0)
mp.legend(["trajectoire C1","C1","Trajectoire C2","C2","A","B"])
mp.show()












