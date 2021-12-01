#!/usr/bin/env python
from __future__ import print_function
import unittest
import pygsl
from pygsl import bspline
import numpy as np

# helper functions to evaluate cubic b-splines (and their derivatives)
# for case with 3 uniformly spaced breakpoints on [-1,1]
def B04(x):
    b = np.zeros(x.shape)
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = -xp*xp*xp
    return b

def Bp04(x):
    b = np.zeros(x.shape)
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = -3.0*xp*xp
    return b

def B14(x):
    b = 0.25*(1-x)*(1-x)*(1-x)
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = 1.75*xp*xp*xp + 0.75*xp*xp - 0.75*xp + 0.25
    return b

def Bp14(x):
    b = -0.75*(1-x)*(1-x)
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = 5.25*xp*xp + 1.5*xp - 0.75
    return b

def B24(x):
    b = x*x*x - 1.5*x*x + 0.5
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = -xp*xp*xp - 1.5*xp*xp + 0.5
    return b

def Bp24(x):
    b = 3.0*x*x - 3.0*x
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = -3.0*xp*xp - 3.0*xp
    return b

def B34(x):
    b = -1.75*x*x*x + 0.75*x*x + 0.75*x + 0.25
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = 0.25*(xp+1)*(xp+1)*(xp+1)
    return b

def Bp34(x):
    b = -5.25*x*x + 1.5*x + 0.75
    ind = (x<0.0)
    xp = x[ind]
    b[ind] = 0.75*(xp+1)*(xp+1)
    return b

def B44(x):
    b = x*x*x;
    ind = (x<0.0)
    b[ind] = 0
    return b

def Bp44(x):
    b = 3.0*x*x;
    ind = (x<0.0)
    b[ind] = 0
    return b


class BsplineTest(unittest.TestCase):
    def setUp(self):
        # Initialize a simple bspline basis
        self.N = 5      # total number of dofs
        self.k = 4      # bspline order (cubic)
        self.Nb = self.N-self.k+2 # number of break points

        xbreak = np.linspace(-1,1,self.Nb) # breakpoints

        # initialize bspline class and set knots based on breakpoints
        self.bw = bspline.bspline(self.k,self.Nb)
        self.bw.knots(xbreak)

        # expected knot locations and greville abscissa
        self.texpect = np.array([-1, -1, -1, -1, 0, 1, 1, 1, 1])
        self.gexpect = np.array([-1, -2./3, 0, 2./3, 1])

        # set a tolerance for these tests
        self.tol = 5*np.finfo(float).eps

    def test_knots(self):
        # check that the knots are what we expect
        t = self.bw.get_internal_knots()
        terr = self.texpect - t
        assert(t.shape[0] == self.N+self.k)
        assert(np.max(np.abs(terr) < self.tol))

    def test_greville(self):
        # check greville abscissa are what we expect
        g = self.bw.greville_abscissa_vector()

        gerr = self.gexpect - g

        assert(g.shape[0] == self.N)
        assert(np.max(np.abs(gerr) < self.tol))

    def test_eval(self):
        # check that eval gives expected results
        x = np.linspace(-1,1,129)
        B04e = B04(x)
        B14e = B14(x)
        B24e = B24(x)
        B34e = B34(x)
        B44e = B44(x)

        B = self.bw.eval_vector(x)

        err0 = B[:,0] - B04e
        err1 = B[:,1] - B14e
        err2 = B[:,2] - B24e
        err3 = B[:,3] - B34e
        err4 = B[:,4] - B44e

        assert(np.max(np.abs(err0)) < self.tol)
        assert(np.max(np.abs(err1)) < self.tol)
        assert(np.max(np.abs(err2)) < self.tol)
        assert(np.max(np.abs(err3)) < self.tol)
        assert(np.max(np.abs(err4)) < self.tol)

    def test_deriv_eval(self):
        # check that deriv_eval gives expected results
        x = np.linspace(-1,1,129)
        B04e = B04(x); Bp04e = Bp04(x);
        B14e = B14(x); Bp14e = Bp14(x);
        B24e = B24(x); Bp24e = Bp24(x);
        B34e = B34(x); Bp34e = Bp34(x);
        B44e = B44(x); Bp44e = Bp44(x);

        B = self.bw.deriv_eval_vector(x,1)

        err0 = B[:,0,0] - B04e; errp0 = B[:,0,1] - Bp04e;
        err1 = B[:,1,0] - B14e; errp1 = B[:,1,1] - Bp14e;
        err2 = B[:,2,0] - B24e; errp2 = B[:,2,1] - Bp24e;
        err3 = B[:,3,0] - B34e; errp3 = B[:,3,1] - Bp34e;
        err4 = B[:,4,0] - B44e; errp4 = B[:,4,1] - Bp44e;

        assert(np.max(np.abs(err0)) < self.tol)
        assert(np.max(np.abs(err1)) < self.tol)
        assert(np.max(np.abs(err2)) < self.tol)
        assert(np.max(np.abs(err3)) < self.tol)
        assert(np.max(np.abs(err4)) < self.tol)

        assert(np.max(np.abs(errp0)) < self.tol)
        assert(np.max(np.abs(errp1)) < self.tol)
        assert(np.max(np.abs(errp2)) < self.tol)
        assert(np.max(np.abs(errp3)) < self.tol)
        assert(np.max(np.abs(errp4)) < self.tol)


if __name__ == '__main__':
    unittest.main()
