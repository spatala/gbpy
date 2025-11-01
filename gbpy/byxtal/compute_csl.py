############################################################
### sympy routines for computing the CSL lattice      ###
############################################################

import numpy as np
from sympy import Matrix, numer
from sympy.polys.matrices import DomainMatrix
from . import smith_normal_form as snf

def compute_csl_grimmer(A, sig_num, sz):
   """
   """
   A = Matrix(np.array(A, dtype='int64'))
   # D, U, V = A.smith_form()
   U, D, V = snf.smith_normal_form(A)


   l_csl_p1 = 0*A
   T0 = D/sig_num
   # l_p1n_p1 = U.inverse()
   l_p1n_p1 = U.inv()
   l_csl_p1[:, 0] = l_p1n_p1[:, 0]
   l_csl_p1[:, 1] = (numer(T0[1, 1]))*l_p1n_p1[:, 1]

   if sz == 3:
      l_csl_p1[:, 2] = T0[2, 2]*l_p1n_p1[:, 2]

   l_csl_p1 = compute_lll(l_csl_p1)
   return l_csl_p1


def compute_lll(A):
    """
    """
    M0 = Matrix(np.array(A, dtype='int64'))
    # (M0.transpose().to_DM().lll().to_Matrix()).transpose()
    M1 = M0.T
    dM = DomainMatrix.from_Matrix(M1)
    M2 = dM.lll()
    M2 = M2.to_Matrix()
    M3 = M2.T
    return np.array(M3, dtype='int64')



