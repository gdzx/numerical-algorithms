import numpy as np
import matplotlib.pyplot as mp
import unittest

# QUESTION 1
def rp(x, p):
	if (p <= 0):
		return None
	i = 0
	sign = 0

	if x < 0:
		sign = 1
		x = -x
	elif x == 0:
		return 0

	while x <= 1:
		x *= 10
		i -= 1

	while x > 1:
		x /= float(10)
		i += 1

	x *= 10**p
	x = np.rint([x])[0]

	return x*pow(10, i - p)*pow(-1, sign)

# TESTS
class TestRpFunction(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(rp(0,5), 0)
        self.assertEqual(rp(-0,2), 0)
    def test_negative(self):
    	self.assertEqual(rp(-1, 2), -1)
    	self.assertEqual(rp(-1.49, 2), -1.5)
    def test_positive(self):
    	self.assertEqual(rp(10.49, 3), 10.5)
    	self.assertEqual(rp(0.0050, 3), 0.005)
    	self.assertEqual(rp(1, 3), 1)
    def test_limits(self):
    	self.assertEqual(rp(1,-5), None)

# QUESTION 2
def add(x, y, p):
	return rp(x+y,p)

def prod(x, y, p):
	return rp(x*y,p)

# TESTS
class TestOperationsFunction(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(0, 5, 2), 5)
        self.assertEqual(add(1.44, 0.02, 2), 1.5)
        self.assertEqual(add(1.3, 0.01, 2), 1.3)
        self.assertEqual(add(1.3, -1.3, 2), 0.0)
    def test_prod(self):
    	self.assertEqual(prod(0, 10, 2), 0.0)
    	self.assertEqual(prod(1, 10, 2), 10)
    	self.assertEqual(prod(1, 1.5, 2), 1.5)

# QUESTION 3
def deltaS(x, y, p):
	m = add(x, y, p)
	r = x + y
	if m != 0:
		return abs((m-r)/m)
	return 0; # Eventuellement en quelques termes dont on ne veut pas tenir compte

def deltaP(x, y, p):
	m = prod(x, y, p)
	r = x * y
	return abs((m-r)/m)

# TESTS
class TestOperationsFunction(unittest.TestCase):
    def test_deltaS(self):
        self.assertEqual(deltaS(2.0, 42, 2), 0)
    def test_deltaP(self):
    	self.assertEqual(deltaP(2.0, 42, 2), 0)

# QUESTION 4
## SOMME
x = np.pi
p = 2
rg = 2000
ts = [i for i in range(rg)]
ss = [deltaS(x, i, p) for i in range(rg)]

mp.xlabel("Valeur de y")
mp.ylabel("Erreur relative")
mp.title("Erreur relative de la somme avec p = %i" % (p))
mp.plot(ts, ss)

x = 1/np.e;
tp = [i for i in range(1,rg)];
sp = [deltaS(x, i, p) for i in range(1,rg)];
mp.plot(tp, sp);
mp.legend(['pi','1/e'])

mp.savefig("Somme")

## PRODUIT
mp.clf();
x = np.pi;
p = 2;
rg = 2000;
tp = [i for i in range(1,rg)];
sp = [deltaP(x, i, p) for i in range(1,rg)];

mp.xlabel("Valeur de y");
mp.ylabel("Erreur relative");
mp.title("Erreur relative du produit avec p = %i" % (p));
mp.plot(tp, sp);

x = 1/np.e;
tp = [i for i in range(1,rg)];
sp = [deltaP(x, i, p) for i in range(1,rg)];
mp.plot(tp, sp);

mp.legend(['pi','1/e'])

mp.savefig("Produit")

# QUESTION 5
def ln2(p):
	"Methode inefficace de sommation sur p decimales"
	y = 0;
	i = 10**p;
	z = 0;
	t = 0.5;
	while i > 0 and t != 0:
		t = (-1)**(i+1)*prod(1, 1/float(i), p)
		z += deltaS(y, t, p) #+ deltaP(1, 1/float(i), p)
		y = add(y, t, p)
		i -= 1;
	return [rp(y,p), z];

def ln2_1(p):
	"Methode plus efficace de sommation (sur les flottants Python)"
	y = 0;
	i = 10**p;
	t = 0;
	while i > 0:
		t = pow(-1,i+1) / float(i)
		y += t;
		i -= 1;
	return y;

def ln2_aitken(n):
	"Methode d'Aitken (sur les flottants Python)"
	s = 0
	i = 1
	S = []
	while i <= n + 1:
		s += (-1)**(i+1) / float(i)
		S.append(s)
		i += 1
	T = [2, 3, 4]
	while len(T) >= 3:
		T = []
		while len(S) >= 3:
			s_1 = S.pop(0)
			s0 = S[0]
			s1 = S[1]
			T.append((s_1*s1-s0*s0)/(s1-2*s0+s_1))
		S = T
	return T[0]

# Graphes
mp.clf()
rg = 10
tp = [i for i in range(4,rg)]
sp = [ln2_aitken(i) for i in range(4,rg)]
sp1 = [np.log(2) for i in range(4,rg)]

mp.xlabel("Nombre de sommes partielles considerees")
mp.ylabel("Valeur approximee")
mp.title("Aitken pour ln(2)")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Aitken', 'ln(2)'])
mp.savefig("Aitken")

mp.clf()
rg = 6
tp = [i for i in range(1,rg)]
sp = [ln2(i)[0] for i in range(1,rg)]
sp1 = [np.log(2) for i in range(1,rg)]
mp.xlabel("Precision p en decimales")
mp.ylabel("Approximation de ln(2)")
mp.title("Sommation inefficace pour ln(2)")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Approximation', 'ln(2)'])
mp.savefig("sommation")

mp.clf()
rg = 6
tp = [i for i in range(1,rg)]
sp = [ln2(i)[1] for i in range(1,rg)]
mp.xlabel("Precision p en decimales")
mp.ylabel("Erreur relative cumulee sur le resultat")
mp.title("Sommation inefficace pour ln(2)")
mp.plot(tp, sp)
mp.savefig("sommation_erreur")

# Tests call
if __name__ == '__main__':
    unittest.main()