from sympy import Matrix, gcdex
from . import integer_manipulations as int_man
import numpy as np;
import numpy.linalg as nla
from . import reduce_po_lat as rpl

# check_basis_equi
# check_basis_def

# def reduce_po_lat(l_dsc_p, l_p_po, tol1=1e-6):
#     N0, _ = int_man.int_mult(l_dsc_p);
#     l_po_p = l_p_po.inv();
#     l_dsc_po = l_p_po*l_dsc_p;

#     N1, imat1 = int_man.int_mult(l_dsc_po, tol1);
#     ### Check that imat1 is indeed an integer matrix
#     imat1 = np.array(imat1, dtype='double')
#     cond1 = int_man.check_int_mat(imat1, tol1);
#     ### Check that l_dsc_po == imat1/N1
#     cond2 = (np.max(np.abs(l_dsc_po - imat1/N1)) < tol1);
#     if cond1 and cond2:
#         imat2 = (np.around(imat1)).astype(int)
#         # imat2 = Matrix(imat1)
#         lll_imat2 = lll_reduction(imat2);
#         cond3 = check_basis_equi(imat2, lll_imat2);
#         if cond3:
#             lll_dsc_po = lll_imat2/N1;
#             cond4 = check_basis_equi(l_dsc_po, lll_dsc_po);
#             if cond4:
#                 lll_dsc_p = l_po_p*lll_dsc_po;
#                 if (N0 == 1):
#                     N1, lll_dsc_p = int_man.int_mult(lll_dsc_p);
#                     if (N1 != 1):
#                         raise Exception("Matrix in reference frame 'p' is not an integer matrix.")
#                     else:
#                         return lll_dsc_p;
#                 else:
#                     return lll_dsc_p;
#             else:
#                 raise Exception("Lattice Reduction did not work.")
#         else:
#             raise Exception("Lattice Reduction did not work.")
#     else:
#         raise Exception("Lattice Reduction did not work.")
# # -----------------------------------------------------------------------------------------------------------

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
            l_p2_T = l_p2.transpose()
            # l_p2_inv = (((l_p2_T*l_p1).inv())*l_p2.T)
            l_p2_inv = nla.inv((l_p2_T.dot(l_p1))).dot(l_p2.T)
            int_mat = (l_p2_inv).dot(l_p1)
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
    exec_str = '/compute_lll.py'
    inp_args = {}
    inp_args['mat'] = int_mat
    lll_int_mat = rpl.call_sage_math(exec_str, inp_args)
    return lll_int_mat


