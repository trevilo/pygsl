import numpy as np
from pygsl import bspline as bsp
import matplotlib.pyplot as plt


def plot_solution(xplt, uplt, uexact):
    import matplotlib.pyplot as plt
    plt.plot(xplt, uplt, 'b-' , lw=2, label='Discret soln')
    plt.plot(xplt, uexact, 'r--', lw=2, label='Exact soln')
    plt.legend()
    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.show()
    plt.close()

    plt.plot(xplt, uplt-uexact, 'b-', lw=2)
    plt.show()

def solve_linear_reaction_diffusion(k, N, s=20.0, makeplot=False):
    nbreak = N-k+2;
    basis = bsp.bspline(k,nbreak)
    xbreak = np.linspace(-1,1,nbreak)

    # Set knots based on breakpoints
    basis.knots(xbreak)

    # Use Greville abscissa as collocations points
    xcol = basis.greville_abscissa_vector()

    # For the required operators
    B = basis.deriv_eval_vector(xcol, 2)

    B0 = np.copy(B[:,:,0])
    B2 = np.copy(B[:,:,2])

    A = -B2 + s*B0;

    # Update operator for Dirichlet BCs
    A[0,:] = np.zeros(N)
    A[-1,:] = np.zeros(N)

    A[0,0] = 1.0
    A[-1,-1] = 1.0

    # Form the RHS
    b = np.zeros(N)
    b[0] = 1.0
    b[-1] = 0.0

    # Solve the system: U = coefficients representing solution
    # i.e., the solution in physical space is u(x) = \sum_i U[i]*B_i(x)
    U = np.linalg.solve(A,b)

    # compute the exact solution
    Abc = np.zeros((2,2))
    Abc[0,0] = np.exp(-np.sqrt(s))
    Abc[0,1] = np.exp( np.sqrt(s))
    Abc[1,0] = np.exp( np.sqrt(s))
    Abc[1,1] = np.exp(-np.sqrt(s))

    bbc = np.array([1.0, 0.0])
    C = np.linalg.solve(Abc, bbc)

    xplt = np.linspace(xbreak[0], xbreak[-1], 1025)
    uexact = C[0]*np.exp(np.sqrt(s)*xplt) + C[1]*np.exp(-np.sqrt(s)*xplt)

    # compute solution in physical space
    Bplt = basis.eval_vector(xplt)
    uplt = Bplt @ U

    if (makeplot):
        plot_solution(xplt, uplt, uexact)

    # return approx of L^{\infty} error
    return np.max(np.abs(uplt - uexact))

if __name__ == '__main__':

    import pygsl
    pygsl.set_debug_level(0)

    kvec = np.array([3, 4, 5, 6, 7, 8], dtype=np.int)
    rvec = np.array([2, 2, 4, 4, 6, 6])

    print("Spline order (k)  Obs. rate  Exp. rate")
    for k, r in zip(kvec.astype(int), rvec):
        k = np.int(k) # why is this necessary?  already an int right?

        err = np.zeros(4)
        err[0] = solve_linear_reaction_diffusion(k, 17);
        err[1] = solve_linear_reaction_diffusion(k, 33);
        err[2] = solve_linear_reaction_diffusion(k, 65);
        err[3] = solve_linear_reaction_diffusion(k, 129);

        rate = np.log(err[-1]/err[-2])/np.log(0.5)

        print("{0:16d} {1:10.3f} {2:10.3f}".format(k, rate, r))




