#!/usr/bin/env sage

import sys
from sage.all import *
import numpy as np
import numpy.linalg as nla
import pickle as pkl
import sage_util_funcs as suf
import integer_manipulations as iman


def reduce_po_lat(l_csl_p, l_p_po, tol):
    """
    """
    l_p_po = np.array(l_p_po, dtype='double')
    l_csl_po = l_p_po.dot(l_csl_p)
    lInt_csl_po, m1 = iman.int_approx(l_csl_po, tol)

    lllInt_csl_po = suf.compute_lll(lInt_csl_po)
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
        Tmat = Matrix(np.dot(A2inv, A1))

    cond1 = iman.check_int_mat(Tmat, 1e-12)
    if cond1:
        Tmat1 = np.array(np.around(np.array(Tmat, dtype='double')), dtype='int64')
        return Tmat1
    else:
        raise Exception("Tmat is not an integer matrix.")


args = len(sys.argv)
pkl_dir = sys.argv[1]
pkl_in_fl = sys.argv[2]
pkl_name = pkl_dir + pkl_in_fl

jar = open(pkl_name, "rb" )
csl_inp_mats = pkl.load(jar)
jar.close()

Nmats = csl_inp_mats['N']
Dmats = csl_inp_mats['D']
l_p_po = csl_inp_mats['l_p_po']
tol1 = csl_inp_mats['tol']

nsz = np.shape(Nmats)[0]
l_csl_p_mats = np.zeros((nsz,3,3), dtype='int64')

for ct1 in range(nsz):
   Nmat = Nmats[ct1]
   Dmat = Dmats[ct1]

   sig_num = int(np.unique(Dmat)[0])
   Sz = np.shape(Nmat)
   l_csl1_p = suf.compute_csl_grimmer(Nmat, sig_num, Sz[0])
   l_csl2_csl1 = reduce_po_lat(l_csl1_p, l_p_po, tol1)

   l_csl2_p = l_csl1_p.dot(l_csl2_csl1)
   l_csl_p_mats[ct1, :, :] = l_csl2_p


pkl_out_fl = sys.argv[3]
pkl_name = pkl_dir+pkl_out_fl
jar = open(pkl_name, 'wb')
pkl.dump(l_csl_p_mats, jar)
jar.close()

