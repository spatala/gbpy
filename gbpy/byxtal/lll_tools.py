from sympy import Matrix, gcdex
from . import integer_manipulations as int_man
import numpy as np;
import numpy.linalg as nla


def check_basis_equi(l_p1, l_p2, tol1 = 1e-10):
    """
    Function 

    Parameters
    ----------------
    l_p1: numpy.arrays
    l_p2: numpy.arrays
    tol1: float
        Tolerance with default value 1e-10

    Returns
    ----------

    """
    sz = np.shape(l_p1)
    if sz[0] == sz[1]:
        int_mat = (nla.inv(l_p1)).dot(l_p2)
        cond1 = int_man.check_int_mat(int_mat, tol1)
        tol2 = tol1*np.max(np.abs(int_mat))
        cond2 = (abs(abs(nla.det(int_mat)) - 1) < tol2)
        if (cond1 and cond2):
            int_mat = (nla.inv(l_p2)).dot(l_p1)
            cond1 = int_man.check_int_mat(int_mat, tol1)
            tol2 = tol1*np.max(np.abs(int_mat))
            cond2 = ( abs(abs( nla.det(int_mat)  ) - 1) < tol2)
            return (cond1 and cond2)
        else:
            return False
    else:
        l_p1_T = l_p1.transpose()
        l_p1_inv = ((nla.inv(l_p1_T.dot(l_p1)))).dot(l_p1_T)
        int_mat = (l_p1_inv).dot(l_p2)
        cond1 = int_man.check_int_mat(int_mat, tol1)
        tol2 = tol1*np.max(np.abs(int_mat))
        cond2 = ( abs(abs(nla.det(int_mat)) - 1) < tol2)
        if (cond1 and cond2):
            # l_p2_inv = (((l_p2.T*l_p1).inv())*l_p2.T)
            # l_p2_inv = np.dot((nla.inv(np.dot((l_p2.T, l_p1)))), l_p2.T)
            l_p2_inv = (nla.inv(l_p2.T.dot(l_p1))).dot(l_p2.T)

            int_mat = np.dot(l_p2_inv, l_p1)
            cond1 = int_man.check_int_mat(int_mat, tol1)
            tol2 = tol1*np.max(np.abs(int_mat))
            cond2 = ( abs(abs(nla.det(int_mat)) - 1) < tol2)
            return (cond1 and cond2)
        else:
            return False


def check_basis_def(l_p1, l_p2, tol1 = 1e-10):
    """
    If l_p2 is defined in l_p1
    (l_p1.inv())*l_p2 is an integer matrix

    Parameters
    ----------------
    l_p1: numpy.arrays
       
    l_p2: numpy.arrays
    tol1: float
        Tolerance with default value 1e-10

    Returns
    ----------
    cond1:
    """
    sz = np.shape(l_p1)

    if sz[0] == sz[1]:
        int_mat = (nla.inv(l_p1)).dot(l_p2)
        cond1 = int_man.check_int_mat(int_mat, tol1)
        return cond1
    else:
        l_p1_T = l_p1.transpose()
        l_p1_inv = ((nla.inv(l_p1_T.dot(l_p1)))).dot(l_p1_T)
        int_mat = (l_p1_inv).dot(l_p2)
        cond1 = int_man.check_int_mat(int_mat, tol1)
        return cond1


def lll_reduction(int_mat):
    """
    Function calculated the lll reduction.
    Parameters
    ----------------
    int_mat:

    Returns
    ----------
    lll_int_mat:
    """
    M = Matrix(int_mat)
    M_lll = (M.transpose().to_DM().lll().to_Matrix()).transpose()
    return np.array(M_lll, dtype='int64')



