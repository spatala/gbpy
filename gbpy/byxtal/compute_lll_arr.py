#!/usr/bin/env sage

import sys
from sage.all import *
import numpy as np
import integer_manipulations as iman
import numpy.linalg as nla
import pickle as pkl
import sage_util_funcs as suf

args = len(sys.argv)
pkl_dir = sys.argv[1]
pkl_in_fl = sys.argv[2]
pkl_name = pkl_dir + pkl_in_fl

jar = open(pkl_name, "rb" )
lll_inp_mats = pkl.load(jar)
jar.close()

l_sig_p_mats = lll_inp_mats['l_sig_p']
l_p_po = lll_inp_mats['l_p_po']
tol1 = lll_inp_mats['tol']

nsz = np.shape(Nmats)[0]
lll_sig_p_mats = np.zeros((nsz,3,3), dtype='int64')

for ct1 in range(nsz):
    l_sig1_p = l_sig_p_mats[ct1]
    l_sig2_sig1 = reduce_po_lat(l_csl1_p, l_p_po, tol1)
    l_sig2_p = l_sig1_p.dot(l_sig2_sig1)
    lll_sig_p_mats[ct1, :, :] = l_sig2_p

pkl_out_fl = sys.argv[3]
pkl_name = pkl_dir+pkl_out_fl
jar = open(pkl_name, 'wb')
pkl.dump(lll_sig_p_mats, jar)
jar.close()
