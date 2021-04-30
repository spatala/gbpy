# Authors: Arash Dehghan Banadaki <adehgha@ncsu.edu>, Srikanth Patala <spatala@ncsu.edu>
# Copyright (c) 2015,  Arash Dehghan Banadaki and Srikanth Patala.
# License: GNU-GPL Style.
# How to cite GBpy:
# Banadaki, A. D. & Patala, S. "An efficient algorithm for computing the primitive
# bases of a general lattice plane",
# Journal of Applied Crystallography 48, 585-588 (2015). doi:10.1107/S1600576715004446

import numpy as np
from math import gcd
from . import integer_manipulations as int_man
from . import find_csl_dsc as fcd
from . import lll_tools as lt
from . import reduce_po_lat as rpl
from .tools import Col, extgcd
import numpy.linalg as nla


def check_2d_csl(l_pl1_g1, l_pl2_g1, l_csl_g1):
    """
    The function checks whether or not the CSL basis may be expressed
    as a linear integer combination of the plane bases of planes 1 and 2

    Parameters
    ----------------
    l_pl1_g1: numpy.arrays
        Basis vectors for plane 1 in the g1 reference frame
    l_pl2_g1: numpy.arrays
        Basis vectors for plane  2 in the g1 reference frame
    l_csl_g1: numpy.array
        CSL basis vector in the g1 reference frame

    Returns
    ------------
    """

    ct = 1
    str1 = 'Check ' + str(ct) + ' : --> '

    if lt.check_basis_def(l_pl1_g1, l_csl_g1):
        str2 = 'YES\n'
        color = 'yel'
    else:
        raise Exception('The 2D CSL does not contain base1.')

    print(str1)
    txt = Col()
    txt.c_prnt(str2, color)
    # --------------------------
    ct += 1
    str1 = 'Check ' + str(ct) + ' : --> '

    if lt.check_basis_def(l_pl2_g1, l_csl_g1):
        str2 = 'YES\n'
        color = 'yel'
    else:
        raise Exception('The 2D CSL does not contain base2.')

    print(str1)
    txt = Col()
    txt.c_prnt(str2, color)


def lbi_dioph_soln(a, b, c):
    """
    Computes the diophantaine solution for the equation ax + by = c

    Parameters
    ----------------
    a: int
        Coefficient of x in equation x + by = c
    b: int
        Coefficient of y in equation x + by = c
    c: int
        Constant value in equation x + by = c

    Returns
    ------------
    int_soln: int
        Diophantaine solution for the equation ax + by = c
    """
    k = abs(gcd(a, b))
    if np.fmod(c, k) != 0.0:
        raise Exception('No Solutions Exist')

    mult = c / k

    # b1 = extended_euclid(a, b) ####<----------------- must be verified
    b1 = np.zeros(2)
    b1[0], b1[1], _ = extgcd(a, b)

    x1 = mult * b1[0]
    x2 = mult * b1[1]

    if a*x1 + b*x2 - c != 0:
        raise Exception('Something wrong with the Algorithm')

    tk = -(b-a)/(a**2 + b**2)
    tk_mat = np.array([[np.ceil(tk)], [np.floor(tk)]])
    x_sol = np.array([[x1], [x1]]) + tk_mat * b
    y_sol = np.array([[x2], [x2]]) - tk_mat * a
    sol_mag = np.power(x_sol, 2) + np.power(y_sol, 2)
    ind = np.where(sol_mag == min(sol_mag))[0]
    int_soln = [x_sol[ind[0]][0], y_sol[ind[0]][0]]
    return int_soln


def compute_basis_vec(d_eq):
    """
    The function computes y1, y2, y3 such that h(y1) + k(y2) + l(y3) = 0
    and modulus of y1 is a minimum

    Parameters
    -----------------
    d_eq: numpy.array or list
        The size is 3 and dimension 1
        h = d_eq[0], k = d_eq[1], l = d_eq[2]

    Returns
    ------------
    np.array([y1, y2, y3]): numpy.array
        Variable
    """
    hp = d_eq[0]
    kp = d_eq[1]
    lp = d_eq[2]
    # Find the minimum y1 such that y2 and y3 are solutions of the equation.
    # kp*y2 + lp*y3 = -hp*y1 (Diaphontane Equation).
    # Solutions exist if gcd(kp,lp) is a multiple of hp*y1
    cond = 0
    y1 = 1
    while cond == 0:
        if np.fmod(hp * y1, gcd(kp, lp)) == 0:
            cond = 1
        else:
            y1 += 1

    # Diophantine Equation: ax + by = c
    # To solve: f = kp*x + lp*y + hp*m = 0
    avar = kp
    bvar = lp
    cvar = -hp * y1
    int_soln = lbi_dioph_soln(avar, bvar, cvar)
    y2 = int_soln[0]
    y3 = int_soln[1]
    if (kp*y2 + lp*y3 + hp*y1) != 0:
        raise Exception('Error with Diophantine solution')

    # if np.ndim(y1) > 0:
    #     y1 = y1[0]
    # if np.ndim(y2) > 0:
    #     y2 = y2[0]
    # if np.ndim(y3) > 0:
    #     y3 = y3[0]
    return np.array([y1, y2, y3])


def bp_basis(miller_ind):
    """
    The function computes the primitve basis of the plane if the
    boundary plane indices are specified

    Parameters
    ---------------
    miller_ind: numpy.array
        Miller indices of the plane (h k l)

    Returns
    -----------
    l_pl_g1: numpy.array
        The primitive basis of the plane in 'g1' reference frame
    """
    # If *miller_inds* are not integers or if the gcd != 1
    # miller_ind = int_man.int_finder(miller_ind)
    if (np.ndim(miller_ind) == 2):
        Sz = np.shape(miller_ind)
        if ((Sz[0] == 1) or (Sz[1] == 1)):
            miller_ind = miller_ind.flatten()
        else:
            raise Exception("Wrong Input Type.")
    h = miller_ind[0]
    k = miller_ind[1]
    l = miller_ind[2]

    if h == 0 and k == 0 and l == 0:
        raise Exception('hkl indices cannot all be zero')
    else:
        if h != 0 and k != 0 and l != 0:
            gc_f1_p = gcd(k, l)
            bv1_g1 = np.array([[0], [-l / gc_f1_p], [k / gc_f1_p]])
            bv2_g1 = compute_basis_vec([h, k, l])
            bv2_g1 = bv2_g1.reshape(np.shape(bv2_g1)[0], 1)
        else:
                if h == 0:
                    if k == 0:
                        bv1_g1 = np.array([[1], [0], [0]])
                        bv2_g1 = np.array([[0], [1], [0]])
                    elif l == 0:
                        bv1_g1 = np.array([[0], [0], [1]])
                        bv2_g1 = np.array([[1], [0], [0]])
                    else:
                        gc_f1_p = gcd(k, l)
                        bv1_g1 = np.array([[0], [-l / gc_f1_p],
                                           [k / gc_f1_p]])
                        bv2_g1 = np.array([[1], [-l / gc_f1_p],
                                           [k / gc_f1_p]])
                else:
                    if k == 0:
                        if l == 0:
                            bv1_g1 = np.array([[0], [1], [0]])
                            bv2_g1 = np.array([[0], [0], [1]])
                        else:
                            gc_f1_p = gcd(h, l)
                            bv1_g1 = np.array([[-l / gc_f1_p], [0],
                                               [h / gc_f1_p]])
                            bv2_g1 = np.array([[-l / gc_f1_p], [1],
                                               [h / gc_f1_p]])
                    else:
                        if l == 0:
                            gc_f1_p = gcd(h, k)
                            bv1_g1 = np.array([[-k / gc_f1_p],
                                               [h / gc_f1_p], [0]])
                            bv2_g1 = np.array([[-k / gc_f1_p],
                                               [h / gc_f1_p], [1]])

    #  The reduced basis vectors for the plane
    Tmat = np.array(np.column_stack([bv1_g1, bv2_g1]), dtype='int64')
    l_pl_g1 = lt.lll_reduction(Tmat)
    return l_pl_g1


def pl_density(l_pl_g1, l_g1_go1):
    """
    For a given two-dimensional plane basis, the planar density is
    computed

    Parameters
    ---------------
    l_pl_g1: numpy.array
        Basis vectors of the underlying lattice with respect to the
        primitive reference frame 'g1'
    l_g1_go1: numpy.array
        Basis vectors of the underlying lattice with respect to the
        orthogonal reference frame 'go1'

    Returns
    ----------
    pd: float
        Planar density = (1/area covered by plane basis)
    """
    l_pl_go1 = np.dot(l_g1_go1, l_pl_g1)
    planar_basis_area = nla.norm(np.cross(l_pl_go1[:, 0], l_pl_go1[:, 1]))
    pd = 1.0/planar_basis_area
    return pd


def gb_2d_csl(inds, t_mat, l_p_po, inds_type='miller_index', mat_ref='g1'):
    """
    For a given boundary plane normal 'bp1_p1' and the misorientation
    matrix 't_p1top2_p1', the two-dimensional CSL lattice is computed

    Parameters
    ------------------
    inds: numpy.array
        The boundary plane indices

    inds_type: str
        {'miller_index', 'normal_go', 'normal_g'}

    t_mat: numpy.array
        Transformation matrix from p1 to p2 in 'mat_ref' reference frame

    mat_ref: str
        {'go1', 'p1'}

    Returns
    -----------
    l_2d_csl_p1: numpy.arrays
        ``l_2d_csl_p1`` is the 2d CSL in p1 ref frame.\v
    l_pl1_p1: numpy.arrays
        ``l_pl1_p1`` is the plane 1 basis in p1 ref frame.\v
    l_pl2_p1: numpy.arrays
        ``l_pl2_p1`` is the plane 2 basis in p1 ref frame.\v
    """
    l_po_p = nla.inv(l_p_po)
    l_pR_po = fcd.reciprocal_mat(l_p_po)
    l_po_pR = nla.inv(l_pR_po)

    if inds_type == 'normal_go':
        bp_po1 = np.array(inds)
    elif inds_type == 'miller_index':
        bp_p1R = np.array(inds)
        bp_po1 = l_pR_po.dot(bp_p1R)
    elif inds_type == 'normal_g':
        bp_p1 = np.array(inds)
        bp_po1 = l_p_po.dot(bp_p1)
    else:
        raise Exception('Wrong index type')

    # Get the 2D planar basis for the surface plane in p1
    bp_p1R = l_po_pR.dot(bp_po1)
    miller1_ind, tm1 = int_man.int_approx(bp_p1R, 1e-6)

    l_pl1_p1 = bp_basis(miller1_ind)
    l_sig2_sig1 = rpl.reduce_po_lat(l_pl1_p1, l_p_po, 1e-6)
    l_pl1_p1 = l_pl1_p1.dot(l_sig2_sig1)

    if mat_ref == 'go1':
        l_po2_po1 = np.copy(t_mat)
        l_po1_po2 = nla.inv(l_po2_po1)
        l_p2_p1 = np.dot(l_po_p, np.dot(l_po2_po1, l_p_po))
    elif mat_ref == 'g1':
        l_p2_p1 = np.copy(t_mat)
        l_po2_po1 = l_p_po.dot(l_p2_p1.dot(l_po_p))
        l_po1_po2 = nla.inv(l_po2_po1)
    else:
        raise Exception('Wrong reference axes')

    # Get the 2D planar basis for the surface plane in p2 and p1
    bp_po2 = -l_po1_po2.dot(bp_po1)
    bp_p2R = l_po_pR.dot(bp_po2)
    miller2_ind, tm1 = int_man.int_approx(bp_p2R, 1e-6)
    l_pl2_p2 = bp_basis(miller2_ind)
    l_sig2_sig1 = rpl.reduce_po_lat(l_pl2_p2, l_p_po, 1e-6)
    l_pl2_p2 = l_pl2_p2.dot(l_sig2_sig1)
    l_pl2_p1 = l_p2_p1.dot(l_pl2_p2)

    # Get the 2D planar basis for the GB in csl basis
    l_csl_p1 = fcd.csl_finder(t_mat, l_p_po, 1e-6)
    l_csl_po1 = l_p_po.dot(l_csl_p1)
    l_cslR_po1 = fcd.reciprocal_mat(l_csl_po1)
    l_po1_cslR = nla.inv(l_cslR_po1)
    n_cslR = l_po1_cslR.dot(bp_po1)
    mind_cslR, tm1 = int_man.int_approx(n_cslR, 1e-6)
    l_bpb_csl = bp_basis(mind_cslR)

    # Covert the 2D planar basis for the GB to p1 reference frame
    l_bpb_p1 = l_csl_p1.dot(l_bpb_csl)
    l_sig2_sig1 = rpl.reduce_po_lat(l_bpb_p1, l_p_po, 1e-6)
    l_2d_csl_p1 = l_bpb_p1.dot(l_sig2_sig1)
    return l_2d_csl_p1, l_pl1_p1, l_pl2_p1


def bicryst_planar_den(inds, t_mat, l_g_go, inds_type='miller_index',
                       mat_ref='go1'):
    """
    The function computes the planar densities of the planes
    1 and 2 and the two-dimensional CSL

    Parameters
    ---------------
    inds: numpy.array
        The boundary plane indices.
    inds_type: str
        {'miller_index', 'normal_go', 'normal_g'}
    t_mat: numpy.array
        Transformation matrix from g1 to g2 in go1 (or g1) reference frame.
    mat_ref: str
        {'go1', 'g1'}

    Returns
    -----------
    pl_den_pl1: numpy.array
        The planar density of planes 1.
    pl_den_pl1, pl_den_pl2: numpy.array
        The planar density of planes 2.
    pl_den_csl: numpy.array
        The planare density of the two-dimensional CSL.
    """
    l_g1_go1 = l_g_go
    l_rg1_go1 = fcd.reciprocal_mat(l_g1_go1)
    l_go1_rg1 = nla.inv(l_rg1_go1)

    if inds_type == 'normal_go':
        bp1_go1 = inds
        miller1_inds, tm1 = int_man.int_approx(np.dot(l_go1_rg1, bp1_go1), 1e-6)
    elif inds_type == 'miller_index':
        miller1_inds = inds
    elif inds_type == 'normal_g':
        bp1_g1 = inds
        l_g1_rg1 = np.dot(l_go1_rg1, l_g1_go1)
        miller1_inds, tm1 = int_man.int_approx(np.dot(l_g1_rg1, bp1_g1), 1e-6)
    else:
        raise Exception('Wrong index type')

    if mat_ref == 'go1':
        l_2d_csl_g1, l_pl1_g1, l_pl2_g1 = gb_2d_csl(miller1_inds,
                                                    t_mat, l_g_go,
                                                    'miller_index', 'go1')
    elif mat_ref == 'g1':
        l_2d_csl_g1, l_pl1_g1, l_pl2_g1 = gb_2d_csl(miller1_inds,
                                                    t_mat, l_g_go,
                                                    'miller_index', 'g1')
    else:
        raise Exception('Wrong reference axis type')

    check_2d_csl(l_pl1_g1, l_pl2_g1, l_2d_csl_g1)

    pl_den_pl1 = pl_density(l_pl1_g1, l_g1_go1)
    pl_den_pl2 = pl_density(l_pl2_g1, l_g1_go1)
    pl_den_csl = pl_density(l_2d_csl_g1, l_g1_go1)

    return pl_den_pl1, pl_den_pl2, pl_den_csl
