"""

Functions for a few quite general operations needed by quite a few
other functions.

This file contains general operation functions, functions mapping
information down to a time scale as well as to magnitude and
direction distributions.

Åshild Telle / Simula Research Labratory / 2019

"""


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def perform_xy_operation(A, fn):
    """

    Takes values from A and  performs operation defined in fn.

    Arguments:
        A - numpy array of dimensions X x Y x D, D arbitary
        fn - operation f : x, i, j -> E, E arbitary, but must be of
            a formate convertable to numpy array (e.g. a number,
            a list, ...)

    Returns:
        A numpy array with resulting values, of dimension X x Y x E

    """

    X, Y = A.shape[:2]

    # get shape by performing one operation
    shape = (X, Y) + np.asarray(fn(A[0, 0], 0, 0)).shape

    disp = np.zeros(shape)

    for x in range(X):
        for y in range(Y):
            disp[x, y] = fn(A[x, y], x, y)

    return disp
    

def perform_operation(A, fn, over_time):
    """

    Performs an operation on all values of A, where A is assumed to be
    a X x Y x D numpy array, optionally T x X x Y x D, where
    D can be arbitary (e.g. 2 or 2 x 2).

    Arguments:
        A - numpy array of original data (domain)
        fn - function f((x, y), i, j) -> E, E arbitary but must be of
            a formate convertable to numpy array (e.g. a number, 
            a list, ..)
        over_time - boolean, set to True if operation is to performed
            for a number of time steps (if there is a third dimension
            wrapping X and Y), False otherwise

    Returns:
        Numpy array of shape as specified in 'shape', possibly
            like A, with resulting values

    """

    if(over_time):
        T, X, Y = A.shape[:3]

        # perform one operation to get shape; preallocate memory
        shape = (T, X, Y) + np.asarray(fn(A[0,0,0], 0, 0)).shape
        disp = np.zeros(shape)

        for t in range(T):
            disp[t] = perform_xy_operation(A[t], fn)
    else:
        disp = perform_xy_operation(A, fn)

    return disp


def calc_norm_over_time(data):
    """
    Finds sum of L2 norm of each vector,
    
        n(t) = sum(i, j) sqrt(x_{t,i,j}^2 + y_{t,i,j}^2)

    for t = 0 ... T.
    
    Arguments:
        Data - numpy array, of dimensions T x X x Y x 2

    Returns:
        Sum array - numpy array of dimension T  

    T, X, Y = data.shape[:3]
    """

    T, X, Y = data.shape[:3]
    disp_norm = np.zeros(T)
    
    for t in range(T):
        disp_norm[t] = sum(np.linalg.norm(data[t].reshape(X*Y, 2), axis=1))

    return disp_norm


def calc_max_ind(data):
    """

    Finds the index of the maximum value of disp. If there are multiple
    maxima (same number) the first one will be returned.

    Arguments:
        data - 1D list-like data structure, of dimension T

    Returns
        index corresponding to maximum index, integers in [0, T)

    """

    max_t = (0, data[0])

    for t in range(len(data)):
        if(data[t] > max_t[1]):
            max_t = (t, data[t])

    return max_t[0]



def calc_magnitude(data, over_time):
    """

    Get the norm of the vector for every point (x, y).

    Arguments:
        data - numpy array of dimension X x Y x D, where D is
            arbitary; optionally of dimensions T x X x Y X D
        over_time - boolean value; determines if T dimension
            should be included or not

    """


    f = lambda x, i, j : np.linalg.norm(x)

    shape = data.shape[:-1]

    return perform_operation(data, f, over_time=over_time)


def normalize_values(data, over_time):
    """

    Normalises each non-zero vector.

    Arguments:
        data - T x X x Y x 2 numpy array, original values
        over_time - boolean value; determines if T dimension
            should be included or not

    Returns:
        T x X x Y x 2 numpy array, normalized values

    """

    f = lambda x, i, j: (1/np.linalg.norm(x))*x \
            if np.linalg.norm(x) > 1E-10 else np.zeros(2)

    return perform_operation(data, f, over_time=over_time)



if __name__ == "__main__":

    # unit tests
    data = np.random.rand(3, 3, 3, 2)
    idt_fn = lambda x, i, j : x
    
    assert(perform_xy_operation(data[0], idt_fn) is not None)
    assert(perform_operation(data, idt_fn, over_time=True) is not None)
    assert(calc_norm_over_time(data) is not None)
    assert(calc_max_ind(calc_norm_over_time(data)) is not None)
    assert(calc_magnitude(data, over_time=True) is not None)
    assert(normalize_values(data, over_time=True) is not None)

    print("All checks passed for operations.py")
