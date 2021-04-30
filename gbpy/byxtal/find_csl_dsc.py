# Authors: Arash Dehghan Banadaki <adehgha@ncsu.edu>, Srikanth Patala <spatala@ncsu.edu>
# Copyright (c) 2015,  Arash Dehghan Banadaki and Srikanth Patala.
# License: GNU-GPL Style.
# How to cite GBpy:
# Banadaki, A. D. & Patala, S. "An efficient algorithm for computing the primitive bases
# of a general lattice plane",
# Journal of Applied Crystallography 48, 585-588 (2015). doi:10.1107/S1600576715004446

import numpy as np
from . import integer_manipulations as int_man
import numpy.linalg as nla
from . import reduce_po_lat as rpl

# def find_csl_dsc(l_p_po, T_p1top2_p1, tol1=1e-6, print_check=True):
#     """
#     This function calls the csl_finder and dsc_finder and returns
#     the CSL and DSC basis vectors in 'g1' reference frame.

#     Parameters
#     -----------------
#     L_G1_GO1: numpy array
#         The three basis vectors for the primitive unit cell
#         (as columns) are given with respect to the GO1 reference
#         frame.

#     R_G1ToG2_G1: 3X3 numpy array
#         The rotation matrix defining the
#         transformation in 'G1' reference frame. The subscript 'G1' refers
#         to the primitive unit cell of G lattice.

#     Returns
#     l_csl_g1, l_dsc_g1: numpy arrays
#         The basis vectors of csl and dsc lattices in the g1 reference frame
#     """
#     ########################################################################
#     ## Compute Sigma, and Sigma*T
#     T_p1top2_p1 = np.array(T_p1top2_p1, dtype='double')
#     Sigma = sigma_calc(T_p1top2_p1, tol1)
#     TI_p1top2_p1 = T_p1top2_p1*Sigma
#     cond1 = int_man.check_int_mat(TI_p1top2_p1, tol1)
#     if cond1:
#         TI_p1top2_p1 = (np.around(TI_p1top2_p1)).astype(int)
#     else:
#         raise Exception("TI_p1top2_p1 is not an integer matrix.")
#     ########################################################################


#     ########################################################################
#     # l_csl_p = csl_finder(T_p1top2_p1, Sigma, l_p_po, tol1)
#     l_csl_p = csl_finder(T_p1top2_p1, l_p_po, tol1)
#     check_val1 = check_csl(l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)
#     ########################################################################

#     ########################################################################
#     l_dsc_p = dsc_finder(T_p1top2_p1, l_p_po, tol1)
#     check_val2 = check_dsc(l_dsc_p, l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)
#     ########################################################################

#     ########################################################################
#     print([check_val1, check_val2])
#     if (not(check_val1 and check_val2)):
#         raise Exception("Error in Computing CSL or DSC Lattices.")
#     ########################################################################

#     return l_csl_p, l_dsc_p


def find_csl_dsc(l_p_po, T_p1top2_p1, tol1=1e-6, print_check=True):
    """
    This function calls the csl_finder and dsc_finder and returns
    the CSL and DSC basis vectors in 'g1' reference frame.

    Parameters
    -----------------
    L_G1_GO1: numpy.array
        The three basis vectors for the primitive unit cell
        (as columns) are given with respect to the GO1 reference
        frame.
    R_G1ToG2_G1: numpy.array
        The rotation matrix with size 3*3 defining the
        transformation in 'G1' reference frame. The subscript 'G1' refers
        to the primitive unit cell of G lattice.

    Returns
    l_csl_g1: numpy.array
        The basis vectors of csl lattices in the g1 reference frame
    l_dsc_g1: numpy.array
        The basis vectors of dsc lattices in the g1 reference frame
    """
    # Compute Sigma, and Sigma*T
    T_p1top2_p1 = np.array(T_p1top2_p1, dtype='double')
    Sigma = sigma_calc(T_p1top2_p1, tol1)
    TI_p1top2_p1 = T_p1top2_p1*Sigma
    cond1 = int_man.check_int_mat(TI_p1top2_p1, tol1)
    if cond1:
        TI_p1top2_p1 = (np.around(TI_p1top2_p1)).astype(int)
    else:
        raise Exception("TI_p1top2_p1 is not an integer matrix.")

    # l_csl_p = csl_finder(T_p1top2_p1, Sigma, l_p_po, tol1)
    l_csl_p = csl_finder(T_p1top2_p1, l_p_po, tol1)
    check_val1 = check_csl(l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)

    # l_dsc_p = dsc_finder(T_p1top2_p1, l_p_po, tol1)
    # check_val2 = check_dsc(l_dsc_p, l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)

    print([check_val1])
    if (not(check_val1)):
        raise Exception("Error in Computing CSL or DSC Lattices.")
    return l_csl_p


def csl_finder(T_p1top2_p1, l_p_po, tol1):
    """
    The CSL is computed for the bi-crystal

    Parameters
    ----------------
    TI_p1top2_p1: numpy.array
        Sigma*(transformation matrix)
    l_p_po: numpy.array
        basis vectors (as columns) of the underlying lattice expressed in the
        orthogonal 'po' reference frame
    tol1: int
        Tolerance to use to compute the reduced LLL lattice

    Returns
    ------------
    l_csl_p: numpy.array
        The CSL basis vectors (as columns) expressed in the primitive reference

    Notes
    ---------
    The "Reduced" refer to the use of LLL algorithm to compute a
    basis that is as close to orthogonal as possible.
    (Refer to http://en.wikipedia.org/wiki/Lattice_reduction) for further
    detials on the concept of Lattice Reduction
    """

    ########################################################################
    ## Compute Sigma, and Sigma*T
    T_p1top2_p1 = np.array(T_p1top2_p1, dtype='double')
    Sigma = sigma_calc(T_p1top2_p1, tol1)
    TI_p1top2_p1 = T_p1top2_p1*Sigma
    cond1 = int_man.check_int_mat(TI_p1top2_p1, tol1)
    if cond1:
        TI_p1top2_p1 = (np.around(TI_p1top2_p1)).astype(int)
    else:
        raise Exception("TI_p1top2_p1 is not an integer matrix.")
    ########################################################################

    exec_str = '/compute_csl.py'
    inp_args = {}
    inp_args['mat'] = TI_p1top2_p1
    inp_args['sig_num'] = Sigma
    l_csl1_p = rpl.call_sage_math(exec_str, inp_args)

    l_csl_csl1 = rpl.reduce_po_lat(l_csl1_p, l_p_po, tol1)
    l_csl_p = l_csl1_p.dot(l_csl_csl1)

    l_csl_p = make_right_handed(l_csl_p, l_p_po)
    return l_csl_p


def dsc_finder(L_G2_G1, L_G1_GO1, tol1):
    """
    The DSC lattice is computed for the bi-crystal, if the transformation
    matrix l_g2_g1 is given and the basis vectors of the underlying crystal
    l_g_go (in the orthogonal reference go frame) are known. The following
    relationship is used: **The reciprocal of the coincidence site lattice of
    the reciprocal lattices is the DSC lattice**

    Parameters
    ----------------
    l_g2_g1: numpy.array
        Transformation matrix (r_g1tog2_g1)
    l_g1_go1: numpy.array
        Basis vectors (as columns) of the underlying lattice expressed in the
        orthogonal 'go' reference frame

    Returns
    ------------
    l_dsc_g1: numpy.array
        The dsc lattice basis vectors (as columns) expressed in the g1 reference

    Notes
    ---------
    The "Reduced" refer to the use of LLL algorithm to compute a
    basis that is as close to orthogonal as possible.
    (Refer to http://en.wikipedia.org/wiki/Lattice_reduction) for further
    detials on the concept of Lattice Reduction
    """

    L_GO1_G1 = nla.inv(L_G1_GO1)
    # Reciprocal lattice of G1
    # --------------------------------------------------------------
    L_rG1_GO1 = reciprocal_mat(L_G1_GO1)
    L_GO1_rG1 = nla.inv(L_rG1_GO1)
    # Reciprocal lattice of G2
    # --------------------------------------------------------------
    L_G2_GO1 = L_G1_GO1.dot(L_G2_G1)
    L_rG2_GO1 = reciprocal_mat(L_G2_GO1)

    # Transformation of the Reciprocal lattices
    # R_rG1TorG2_rG1 = L_rG2_G1*L_G1_rG1
    L_rG2_rG1 = L_GO1_rG1.dot(L_rG2_GO1)
    Sigma_star = sigma_calc(L_rG2_rG1, tol1)
    # # Check Sigma_star == Sigma
    # LI_rG2_rG1 = L_rG2_rG1*Sigma_star
    # if int_man.check_int_mat(LI_rG2_rG1, 1e-10):
    #     LI_rG2_rG1 = np.around(np.array(LI_rG2_rG1, dtype='double'))
    #     LI_rG2_rG1 = (np.array(LI_rG2_rG1, dtype='int64'))
    # else:
    #     raise Exception("Not an integer matrix")

    # CSL of the reciprocal lattices
    L_rCSL_rG1 = csl_finder(L_rG2_rG1, L_rG1_GO1, tol1)
    L_rCSL_GO1 = L_rG1_GO1.dot(L_rCSL_rG1)

    L_DSC_GO1 = reciprocal_mat(L_rCSL_GO1)
    L_DSC_G1 = L_GO1_G1.dot(L_DSC_GO1)
    Tmat = np.array(L_DSC_G1*Sigma_star, dtype='double')

    if int_man.check_int_mat(Tmat, tol1):
        Tmat = np.around(Tmat)
        Tmat = np.array(Tmat, dtype='int64')
        L_DSC_G1 = Tmat/Sigma_star
    else:
        raise Exception("DSC*Sigma is not an integer matrix")

    L_DSC1_DSC = rpl.reduce_po_lat(L_DSC_G1, L_G1_GO1, tol1)
    LLL_DSC_G1 = L_DSC_G1.dot(L_DSC1_DSC)

    if int_man.check_int_mat(LLL_DSC_G1*Sigma_star, tol1):
        Tmat = np.array(LLL_DSC_G1*Sigma_star, dtype='double')
        Tmat = np.around(Tmat)
        Tmat = np.array(Tmat, dtype='int64')
        LLL_DSC_G1 = Tmat/Sigma_star
    else:
        raise Exception("DSC*Sigma is not an integer matrix")

    L_DSC_G1 = make_right_handed(LLL_DSC_G1, L_G1_GO1)
    return L_DSC_G1


def check_csl(l_csl_p, l_p1_po, T_p1top2_p1, Sigma, print_val):
    """
    The function checks CSL lattice

    Parameters
    ----------
    l_csl_p : numpy.array
        The CSL basis vectors  in the primitive reference frame.
    l_p_po: numpy array
        The primitive basis vectors of the underlying lattice in the orthogonal
        reference frame.
    T_p1top2_p1: numpy.array
        Sigma*(transformation matrix)
    sigma: float
        Sigma number
    print_val: str
        Print a message

    Returns
    -------
    (cond1 and cond2 and cond3): tuple
        Print message
    """
    l_po_p1 = nla.inv(l_p1_po)

    l_csl_po = l_p1_po.dot(l_csl_p)
    cond1 = int_man.check_int_mat(l_po_p1.dot(l_csl_po), 1e-10)

    l_p2_p1 = np.copy(T_p1top2_p1)
    l_p2_po = l_p1_po.dot(l_p2_p1)
    l_po_p2 = nla.inv(l_p2_po)
    cond2 = int_man.check_int_mat(l_po_p2.dot(l_csl_po), 1e-10)

    Sigma1 = nla.det(l_csl_po) / nla.det(l_p1_po)
    cond3 = (np.abs(Sigma-Sigma1) < 1e-8)

    if print_val:
        if cond1:
            Disp_str = 'l_csl_po is defined in the l_p1_po lattice'
            print(Disp_str)
        if cond2:
            Disp_str = 'l_csl_po is defined in the l_p2_po lattice'
            print(Disp_str)
        if cond3:
            Disp_str = ('V(csl_po)/V(p1_po) = Sigma =  ' + "%0.0f" % (Sigma))
            print(Disp_str)
    return (cond1 and cond2 and cond3)


def check_dsc(l_dsc_p1, l_csl_p1, l_p1_po, l_p2_p1, Sigma, print_val):
    """
    The function checks DSC lattice

    Parameters
    ----------
    l_dsc_p1 : numpy.array
        The DSC basis vectors  in the primitive reference frame of crystal 1.
    l_csl_p1 : numpy.array
        The CSL basis vectors  in the primitive reference frame of crystal 1.
    l_p1_po: numpy.array
        The basis vector of p1 in the orthogonal primitive reference frame.
    l_p2_p1: numpy.array
        Transformation matrix
    sigma: float
        sigma number
    print_val: str
        Print a message

    Returns
    -------
    (cond1 and cond2 and cond3): tuple
        Print message
    """
    l_csl_po = l_p1_po.dot(l_csl_p1)
    l_dsc_po = l_p1_po.dot(l_dsc_p1)
    l_p2_po = l_p1_po.dot(l_p2_p1)

    Tmat1 = (nla.inv(l_dsc_po)).dot(l_p1_po)
    cond1 = (int_man.check_int_mat(Tmat1, 1e-10))

    Tmat1 = (nla.inv(l_dsc_po)).dot(l_p2_po)
    cond2 = (int_man.check_int_mat(Tmat1, 1e-10))

    Tmat1 = (nla.inv(l_dsc_po)).dot(l_csl_po)
    cond3 = (int_man.check_int_mat(Tmat1, 1e-10))

    Tmat1 = l_dsc_p1*Sigma
    cond4 = (int_man.check_int_mat(Tmat1, 1e-10))

    Sigma1 = nla.det(l_p1_po)/nla.det(l_dsc_po)
    cond5 = (np.abs(Sigma-Sigma1) < 1e-8)

    if print_val:
        if cond1:
            Disp_str = 'l_p1_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond2:
            Disp_str = 'l_dsc_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond3:
            Disp_str = 'l_csl_po is defined in the l_dsc_po lattice'
            print(Disp_str)
        if cond4:
            Disp_str = 'l_dsc_po*Sigma is an integer matrix'
            print(Disp_str)
        if cond5:
            Disp_str = ('V(p1_po)/V(dsc_po) = Sigma =  ' + "%0.0f" % (Sigma))
            print(Disp_str)
    return (cond1 and cond2 and cond3 and cond4 and cond5)


def sigma_calc(t_g1tog2_g1, tol1):
    """
    Computes the sigma of the transformation matrix (t_g1tog2_g1)

    Parameters
    ----------------
    t_g1tog2_g1: numpy.array
        Transformation matrix
    tol1: float
        Tolerance

    Returns
    -----------
    int(Sigma): int
        Integer value of the calculated sigma number

    Notes
    ------
    * Suppose T = t_g1tog2_g1
    * if det(T) = det(T^{-1}) then sigma1 = sigma2 is returned (homophase)
    * if det(T) \\neq det(T^{-1}) then max(sigma1, sigma2) is returned (heterophase)
    """
    R = np.copy(t_g1tog2_g1)
    R2 = nla.det(R)*nla.inv(R)

    _, Sigma21 = int_man.int_mult_approx(R, tol1)
    _, Sigma22 = int_man.int_mult_approx(R2, tol1)

    Sigma = int(np.array([Sigma21, Sigma22]).max())
    return int(Sigma)


def reciprocal_mat(l_g_go):
    """
    The reciprocal matrix with reciprocal basis vectors is computed for the
    input matrix with primitve basis vectors

    Parameters
    ----------------
    l_g_go: numpy.array
        The primitive basis vectors b1x, b1y, b1z

    Returns
    -----------
    rl_g_go: numpy.array
        The primitve reciprocal basis vectors
    """
    InMat = np.copy(l_g_go)

    L3 = np.cross(InMat[:, 0], InMat[:, 1]) / np.linalg.det(InMat)
    L1 = np.cross(InMat[:, 1], InMat[:, 2]) / np.linalg.det(InMat)
    L2 = np.cross(InMat[:, 2], InMat[:, 0]) / np.linalg.det(InMat)
    rl_g_go = np.vstack((L1, L2, L3)).T

    return rl_g_go


def make_right_handed(l_csl_p1, l_p_po):
    """
    The function makes l_csl_p1 right handed.

    Parameters
    ----------------
    l_csl_p1: numpy.array
        The CSL basis vectors  in the primitive reference frame of crystal 1.
    l_p_po: numpy.array
        The primitive basis vectors of the underlying lattice in the orthogonal
        reference frame.

    Returns
    -----------
    t1_array: numpy.array
        Right handed array        
    """
    l_csl_po1 = l_p_po.dot(l_csl_p1)
    t1_array = np.array(l_csl_p1, dtype='double')
    t2_array = np.array(l_csl_p1, dtype='double')
    if (nla.det(l_csl_po1) < 0):
        t1_array[:, 0] = t2_array[:, 1]
        t1_array[:, 1] = t2_array[:, 0]
    return t1_array

