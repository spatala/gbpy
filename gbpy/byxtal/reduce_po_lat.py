import subprocess
import os
import inspect
import byxtal
import numpy as np
import numpy.linalg as nla
from . import integer_manipulations as iman

from sympy import Matrix
from sympy.polys.matrices import DomainMatrix

def reduce_po_lat(l_csl_p, l_p_po, tol):
    """
    """
    l_p_po = np.array(l_p_po, dtype='double')
    l_csl_po = l_p_po.dot(l_csl_p)
    lInt_csl_po, m1 = iman.int_approx(l_csl_po, tol)

    M = Matrix(lInt_csl_po)
    lll_reduced_dM = (M.transpose().to_DM().lll().to_Matrix()).transpose()
    lllInt_csl_po = (np.array(lll_reduced_dM)).astype(int)

    Sz = np.shape(lllInt_csl_po)

    if Sz[0] == Sz[1]:
        if nla.det(lllInt_csl_po) < 0:
            if Sz[0] == 3:
                M4 = np.array([[0,1,0],[1,0,0],[0,0,1]])
                lllInt_csl_po = lllInt_csl_po.dot(M4)
            if Sz[0] == 2:
                M4 = Matrix([[0,1],[1,0]])
                lllInt_csl_po = lllInt_csl_po.dot(M4)

        Tmat = ((nla.inv(lInt_csl_po))).dot(lllInt_csl_po)
    else:
        A1 = (np.array(lllInt_csl_po, dtype='int64'))
        A2 = (np.array(lInt_csl_po, dtype='int64'))
        A2inv = np.linalg.pinv(A2)
        Tmat = (np.dot(A2inv, A1))

    cond1 = iman.check_int_mat(Tmat, 1e-12)
    if cond1:
        Tmat1 = np.array(np.around(np.array(Tmat, dtype='double')), dtype='int64')
        return Tmat1
    else:
        raise Exception("Tmat is not an integer matrix.")
