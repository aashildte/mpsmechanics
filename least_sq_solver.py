
"""

Module for finding a least squares solution of a
given linear system

Åshild Telle / Simula Research Labratory / 2018

"""


import numpy as np


class Least_sq_solver:
    def __init__(self, X, Y, xs, ys, disp_x, disp_y):
        self.X, self.Y, self.xs, self.ys, \
            self.disp_x, self.disp_y = \
            X, Y, xs, ys, disp_x, disp_y

    def _get_taylor_polynomial_basis(self, N):
        """

        Generates a list of functions, the function space
        basis given by
            {1, x, 1/2 x^2, ..., 1/(N-1)! x^(N-1)}

        Arguments:
            N - integer value

        Returns:
            List of functions from R to R

        """
        
        return [(lambda x,n=n : (1./np.fac(n))*x**n) \
                for n in range(N-1)]


    def _get_trigonometric_basis(self, N):
        """

        Generates a list of functions, the function space
        basis given by
            {1, sin(x), cos(x), ..., 1/sqrt(N-1)cos((N-1)x)}

        Arguments:
            N - integer value

        Returns:
            List of functions from R to R

        """

        basis = []
    
        basis.append(lambda x : 1)

        for n in range(1, 1 + N//2):
            basis.append(lambda x,n=n : np.sin(n*x))
            basis.append(lambda x,n=n : np.cos(n*x))

        return basis[:N]

    def _generate_tensor_product(self, basis_funs, M, N):
        """

        Generates the tensor product of two function
        basises, i.e. the functions spanning
            span(basis_funs[:M]) x span(basis_funs[:N])
        giving a one-dimensional list of MxN basis
        functions back.

        Arguments:
            basis_funs - at least max(M, N) functions
            from R to R, assumed to span a vector space
            M - integer value
            N - integer value

        Returns:
            List of functions from R to R

        """

        basis = []

        for f1 in basis_funs[:M]:
            for f2 in basis_funs[:N]:
                f = lambda x, y, f1=f1, f2=f2: f1(x)*f2(y)
                basis.append(f)

        return basis
    
    def _get_basis_functions(self, M, N, basis_type):
        """

        Transform basis given as string to a set of
        functions.

        Arguments:
            M - number of functions for x direction
            N - number of functions for y direction
            basis_type - string, must be "trig" or "taylor"

        Returns:
            tensor product of M x N functions from the
                given function basis

        """

        if(basis_type == "trig"):
            funs = self._get_trigonometric_basis(max(M, N))
        elif(basis_type == "taylor"):
            funs = self._get_taylor_polynomial_basis(max(M, N))
        else:
            print("Basis function type not recognized.")
            exit(-1)
        
        return self._generate_tensor_product(funs, M, N)
    
    def solve(self, M, N, basis_type):
        """

        Solve the linear system, finding a least squares
        solution.

        Arguments:
            M - number of functions for x direction
            N - number of functions for y direction
            basis_type - string, must be "trig" or "taylor"
        
        Returns:
            solution for x and y components

        """

        X, Y, xs, ys, disp_x, disp_y = \
            self.X, self.Y, self.xs, self.ys, \
            self.disp_x, self.disp_y

        if(M > X or N > Y):
            print(M, X, N, Y)
            print("Error: Only have ", X, "x", Y, "points")
            exit(-1)

        # define matrix
        funs = self._get_basis_functions(M, N, basis_type)
        
        A = np.zeros((X*Y, M*N))

        for n in range(M*N):
            for i in range(X):
                for j in range(Y):
                    A[i*Y + j][n] = funs[n](xs[i], ys[j])
 
        # solve linear system

        bx, by = disp_x.flatten(), disp_y.flatten()

        coeff_lambdas = np.linalg.lstsq(A, bx, rcond=None)[0]
        coeff_kappas = np.linalg.lstsq(A, by, rcond=None)[0]

        VX = np.dot(A, coeff_lambdas).reshape(X, Y)
        VY = np.dot(A, coeff_kappas).reshape(X, Y)

        return VX, VY

