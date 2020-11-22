import os
import random

import byxtal.tools as gbt
import byxtal.lattice as gbl
import pickle as pkl
import math as mt
import numpy as np
import byxtal.quaternion as gbq
import byxtal.misorient_fz as mfz
import byxtal.disorient_symm_props as dsp
import byxtal.find_csl_dsc as fcd
import byxtal.integer_manipulations as iman
import byxtal.pick_fz_bpl as pfb
import byxtal.bp_basis as bpb
import byxtal.reduce_po_lat as rpl

# from . import reduce_po_lat as rpl
# import byxtal.lll_tools as lt

from sympy.matrices import Matrix, eye, zeros

def order_ptgrp(bp_symm_grp):
    prop_grps = ['C1', 'C2', 'C3', 'C4', 'C6', 'D2', 'D3', 'D4', 'D6', 'D8', 'T', 'O']
    laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Th', 'Oh']
    noncentro_grps = ['Cs', 'S4', 'S6', 'C2v', 'C3v', 'C4v', 'C6v', 'D2d', 'D3h', 'Td']

    if bp_symm_grp == 'C1':
        return 1;
    if bp_symm_grp ==  'C2':
        return 2;
    if bp_symm_grp ==  'C3':
        return 3;
    if bp_symm_grp == 'C4':
        return 4;
    if bp_symm_grp == 'C6':
        return 6;
    if bp_symm_grp == 'D2':
        return 4;
    if bp_symm_grp == 'D3':
        return 6;
    if bp_symm_grp == 'D4':
        return 8;
    if bp_symm_grp == 'D6':
        return 12;
    if bp_symm_grp == 'D8':
        return 16;
    if bp_symm_grp == 'T':
        return 12;
    if bp_symm_grp == 'O':
        return 24;

    if bp_symm_grp == 'Ci':
        return 2;
    if bp_symm_grp ==  'C2h':
        return 4;
    if bp_symm_grp ==  'C3i':
        return 6;
    if bp_symm_grp == 'C4h':
        return 8;
    if bp_symm_grp == 'C6h':
        return 12;
    if bp_symm_grp == 'D2h':
        return 8;
    if bp_symm_grp == 'D3d':
        return 12;
    if bp_symm_grp == 'D4h':
        return 16;
    if bp_symm_grp == 'D6h':
        return 24;
    if bp_symm_grp == 'D8h':
        return 32;
    if bp_symm_grp == 'Th':
        return 24;
    if bp_symm_grp == 'Oh':
        return 48;


    if bp_symm_grp == 'Cs':
        return 2;
    if bp_symm_grp == 'S4':
        return 4;
    if bp_symm_grp == 'S6':
        return 6;
    if bp_symm_grp == 'C2v':
        return 4;
    if bp_symm_grp == 'C3v':
        return 6;
    if bp_symm_grp == 'C4v':
        return 8;
    if bp_symm_grp == 'C6v':
        return 12;
    if bp_symm_grp == 'D2d':
        return 8;
    if bp_symm_grp == 'D3h':
        return 12;
    if bp_symm_grp == 'Td':
        return 24;

def gen_hkl_inds(num1):
    #### (hkl) indices in primitive lattice
    ## h,k,l vary in the range $[-num1,num1]$
    # num1 = 3;
    h1 = np.arange(-num1,num1+1);
    h, k, l = np.meshgrid(h1, h1, h1);
    hkl_inds = (np.vstack((h.flatten(), k.flatten(), l.flatten()))).transpose();
    ### Remove h = k = l = 0
    ind1 = np.where(np.sum(hkl_inds**2, axis=1) == 0)[0][0];
    return np.delete(hkl_inds, ind1, axis=0)

def remove_duplicate_hkl(hkl_inds):
    ### Remove duplicate (h k l) indices that are scaled with respect
    ### to each other
    gcd_hkl=iman.gcd_array(hkl_inds, order='rows')
    gcd_hkl = np.tile(gcd_hkl, (1,3));
    hkl_inds1 = hkl_inds/gcd_hkl;
    return gbt.unique_rows_tol(hkl_inds1);

def conv_hkl_uvecs(hkl_inds2, l_p_po):
    ### Compute the Unit normal vectors in 'po' reference frame.
    l_rp_po = np.array(fcd.reciprocal_mat(l_p_po), dtype='double');

    num1 = np.shape(hkl_inds2)[0];
    norm_uvec = np.zeros((num1,3));
    for ct1 in range(num1):
        hkl1 = hkl_inds2[ct1];
        norm_vec = np.dot(l_rp_po, hkl1);
        norm_uvec[ct1,:] = norm_vec/np.linalg.norm(norm_vec);
    return norm_uvec;

def symm_fz_hkl(l_csl_props, hkl_inds):
    ### Compute all the symmetrically equivalent normals and
    ### keep only those normals that are unique and belong to
    ### the fundamental zone (given by the symmetry point grp)

    # symm_grp_ax = np.eye(3,3); symm_grp = 'O_h';
    l_p_po = l_csl_props['l_csl_po'];
    norm_uvec = conv_hkl_uvecs(hkl_inds, l_p_po);
    symm_grp_ax = l_csl_props['symm_grp_ax'];
    bp_symm_grp = l_csl_props['bp_symm_grp'];
    l_rp_po = np.array(fcd.reciprocal_mat(l_p_po), dtype='double');

    bp_fz_norms_go1, bp_fz_stereo = pfb.pick_fz_bpl(norm_uvec, bp_symm_grp, symm_grp_ax, x_tol=1e-04)
    nv_unq = gbt.unique_rows_tol(bp_fz_norms_go1)
    ################################################################################
    ### Convert to nv_unq to Miller indices (h k l)
    l_po_rp = np.linalg.inv(l_rp_po);
    num1 = np.shape(nv_unq)[0];
    hkl_inds = np.zeros((num1, 3));
    for ct1 in range(num1):
        n1_po = nv_unq[ct1];
        n1_rp = np.dot(l_po_rp,n1_po);
        # T1 = iman.int_finder(n1_rp);
        T1, tm1 = iman.int_approx(n1_rp);
        hkl_inds[ct1,:] = np.array(T1.reshape(1,3), dtype='double');
    ################################################################################
    return hkl_inds;

def compute_hkl_bpb(hkl_inds):
    num1 = np.shape(hkl_inds)[0];
    l_p2_p = np.zeros((num1,3,3)); l_bpb_p = np.zeros((num1,3,2));
    for ct1 in range(num1):
        # print(ct1)
        hkl1 = hkl_inds[ct1]; hkl1 = hkl1.astype(int);
        bp1 = bpb.bp_basis(hkl1);
        a_vec = bp1[:,0]; b_vec = bp1[:,1];
        l_bpb_p[ct1,:,:] = np.copy(bp1);
    return l_bpb_p;

def gen_Acut_bpb(l_bpb_p, l_p_po, r_cut, A_cut):
    """Generates superlattices with a minimum area given by
    A_cut.

    elem = 'cF_Id'; l1 = GBl.Lattice(elem);
    lat_par = l1.lat_params['a']; rCut = lat_par*4; A_cut = rCut**2;

    """
    num2 = np.shape(l_bpb_p)[0];
    l_p_po = np.array(l_p_po, dtype='double');

    l_bpbSig_p_mats = np.zeros((num2,3,2));
    for ct1 in range(num2):
        # print(ct1)
        l_bp_p = l_bpb_p[ct1]; l_bp_po = np.dot(l_p_po, l_bp_p);
        #####################################################################
        ### Area of the 2D unit-cell
        area_bpl = np.linalg.norm(np.cross(l_bp_po[:,0], l_bp_po[:,1]));
        sig_num = np.ceil(A_cut/area_bpl);
        #####################################################################
        ind2 = np.array([], dtype='int64');
        while (np.size(ind2) == 0):
            hnf_mats = sig_hnf_mats(sig_num);
            l_sig_p_mats, l_sig_po_mats = compute_hnf_props(hnf_mats, l_bp_p, l_p_po, 0.01);
            ind2 = ind_min_cost(l_sig_po_mats, r_cut);
            sig_num = sig_num + 1;

        l_bpbSig_p_mats[ct1] = l_sig_p_mats[ind2];

    return l_bpbSig_p_mats;

def sig_hnf_mats(sig_num):
    #####################################################################
    ct2 = 1; sig_facs = [];
    for ct1 in range(int(sig_num)):
        if (sig_num % (ct1+1) == 0):
            sig_facs.append(ct1+1);

    hnf_mats = [];
    for a1 in sig_facs:
        c1 = sig_num/a1;
        for b1 in range(int(c1)):
            hnf_mats.append(np.array([[int(a1),0], [int(b1), int(c1)]]));
    #####################################################################
    return hnf_mats;

def compute_hnf_props(hnf_mats, l_bp_p, l_p_po, tol):
    num_hnf = len(hnf_mats);
    l_sig_p_mats = np.zeros((num_hnf, 3, 2));
    l_sig_po_mats = np.zeros((num_hnf, 3, 2));
    for hct1 in range(num_hnf):
        # print(hct1)
        l_sig_p = np.dot(l_bp_p, hnf_mats[hct1]);
        # l_sig1_p = lll_reduction_bpl_basis(l_sig_p, l_p_po);
        l_sig1_sig = rpl.reduce_po_lat(l_sig_p, l_p_po, tol)
        l_sig1_p = l_sig_p.dot(l_sig1_sig)
        l_sig_p_mats[hct1] = l_sig1_p;
        l_sig_po = np.dot(l_p_po, l_sig1_p);
        l_sig_po_mats[hct1] = l_sig_po;
    return l_sig_p_mats, l_sig_po_mats;

def ind_min_cost(l_sig_po_mats, r_cut):
    #### Can define multiple costs
    num1 = np.shape(l_sig_po_mats)[0];
    ldiffs = np.zeros((num1,));
    langs = np.zeros((num1,));
    l1_norms = np.zeros((num1,));
    l2_norms = np.zeros((num1,));
    for hct1 in range(num1):
        l_sig_po = l_sig_po_mats[hct1];
        l1 = np.linalg.norm(l_sig_po[:,0]);
        l1_norms[hct1] = l1;
        l1_uvec = l_sig_po[:,0]/l1;
        l2 = np.linalg.norm(l_sig_po[:,1]);
        l2_uvec = l_sig_po[:,1]/l2;
        l2_norms[hct1] = l2;
        ldiffs[hct1] = np.abs(l2 - l1);
        ang1 = np.arccos(np.dot(l1_uvec, l2_uvec))*180/np.pi;
        langs[hct1] = ang1

    cond1 = (l1_norms > r_cut);
    cond2 = (l2_norms > r_cut);
    inds2 = np.where((cond1) & (cond2))[0];
    if (np.size(inds2) > 0):
        lcost = ldiffs[inds2] + np.abs(langs[inds2]-90);
        ind1 = np.where(lcost == np.min(lcost))[0];
        return(inds2[ind1[0]]);
    else:
        return inds2;


def gen_hkl_props(l_csl_props, num1):
    l_p_po = l_csl_props['l_csl_po'];
    hkl_inds = gen_hkl_inds(num1);
    hkl_inds2 = remove_duplicate_hkl(hkl_inds);
    hkl_inds = symm_fz_hkl(l_csl_props, hkl_inds2);
    l_bpb_p = compute_hkl_bpb(hkl_inds);
    l_bpb_p = l_bpb_p.astype(int);
    return hkl_inds, l_bpb_p;


