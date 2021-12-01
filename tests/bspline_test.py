#!/usr/bin/env python
from __future__ import print_function
import unittest
import pygsl
from pygsl import bspline
import numpy as np

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


if __name__ == '__main__':
    unittest.main()
