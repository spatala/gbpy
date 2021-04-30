#!/usr/bin/env sage

import numpy as np
from sage.all import *


def compute_csl_grimmer(A, sig_num, sz):
    """
    """
    A = Matrix(np.array(A, dtype='int64'))
    D, U, V = A.smith_form()
    l_csl_p1 = 0*A
    T0 = D/sig_num
    l_p1n_p1 = U.inverse()
    l_csl_p1[:, 0] = l_p1n_p1[:, 0]
    l_csl_p1[:, 1] = (T0[1, 1].numerator())*l_p1n_p1[:, 1]

    if sz == 3:
        l_csl_p1[:, 2] = T0[2, 2]*l_p1n_p1[:, 2]

    l_csl_p1 = compute_lll(l_csl_p1)
    return l_csl_p1


def compute_lll(A):
    """
    """
    M0 = Matrix(np.array(A, dtype='int64'))
    M1 = M0.transpose()
    M2 = M1.LLL()
    M3 = M2.transpose()
    return np.array(M3, dtype='int64')
