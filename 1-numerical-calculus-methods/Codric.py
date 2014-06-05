import numpy as np
import matplotlib.pyplot as mp
import unittest

def frange(x, y, jump):
    "Range for ... floats !"
    while x < y:
        yield x
        x += jump

L = [np.log(1+10**(-i)) for i in range(0,6)]

def ln(x):
    "Fonction logarithme neperien"
    s = 0
    if x < 1:
        x = 1/x
        s = 1
    y=0
    p=1
    for k in range(0,6):
        while (x >= p + p*10**(-k)):
            y = y + L[k]
            p = p + p*10**(-k)
    return (-1)**s*y+(x/p-1)


def exp(x):
    "Fonction exponentielle"
    y=1
    for k in range (0,6):
        while (x >= L[k]):
            x = x - L[k]
            y = y + y * 10**(-k)
    return y + y*x

A = [np.arctan(10**(-i)) for i in range(0, 6)]

def arctan(x):
    "fonction Arctangente"
    y=1
    r=0
    k=0
    sign = 0
    if x < 0:
        x = -x
        sign = 1
    while (k<=5):
        while (x >= y * 10**(-k)):
            xp = x - y * 10**(-k)
            y = y + x * 10**(-k)
            x = xp
            r = r + A[k]
        k = k + 1
    return (-1)**sign * r + (x/y)

def tan(x):
    "fonction tangente"
    n=0
    d=1
    if x < 0:
        x %= np.pi
    for k in range (0,4):
        while (x>=A[k]):
            x = x-A[k]
            nup = n + d * 10**(-k)
            d = d - n * 10**(-k)
            n = nup
    return (n+x*d)/(d-x*n)

# GRAPHES
start = 0.1
rg = 30
step = 0.2
tp = [i for i in frange(start,rg, step)]
sp = [ln(i) for i in frange(start,rg, step)]
sp1 = [np.log(i) for i in frange(start,rg, step)]

mp.xlabel("x")
mp.ylabel("ln(x)")
mp.title("Graphe de la fonction ln()")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Approximation', 'ln()'])
mp.savefig("ln")

mp.clf()
rg = 10
step = 0.5
tp = [i for i in frange(-rg,rg, step)]
sp = [exp(i) for i in frange(-rg,rg, step)]
sp1 = [np.exp(i) for i in frange(-rg,rg, step)]

mp.xlabel("x")
mp.ylabel("exp(x)")
mp.title("Graphe de la fonction exp()")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Approximation', 'exp()'])
mp.savefig("exp")

mp.clf()
rg = 10
step = 0.5
tp = [i for i in frange(-rg,rg,step)]
sp = [arctan(i) for i in frange(-rg,rg,step)]
sp1 = [np.arctan(i) for i in frange(-rg,rg,step)]

mp.xlabel("x")
mp.ylabel("Arctan(x)")
mp.title("Graphe de la fonction Arctan()")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Approximation', 'Arctan()'])
mp.savefig("arctan")

mp.clf()
rg = 10
step = 0.01
tp = [i for i in frange(-rg,rg, step)]
sp = [tan(i) for i in frange(-rg,rg, step)]
sp1 = [np.tan(i) for i in frange(-rg,rg, step)]

mp.xlabel("x")
mp.ylabel("tan(x)")
mp.title("Graphe de la fonction tan()")
mp.plot(tp, sp)
mp.plot(tp, sp1)
mp.legend(['Approximation', 'tan()'])
mp.axis([-10,10,-10,10])
mp.savefig("tan")

# TESTS
class TestFunctions(unittest.TestCase):
    def test_ln(self):
        self.assertEqual(ln(1), 0)
    def test_exp(self):
        self.assertEqual(exp(0), 1)
    def test_arctan(self):
        self.assertEqual(arctan(0), 0)
    def test_tan(self):
        self.assertEqual(tan(0), 0)

if __name__ == '__main__':
    unittest.main()