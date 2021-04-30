##################################################################
## Code to enumerate the properties of the CSL/DSC lattices for
## Sigma misorientations
## The code currently works for cubic lattices.
##################################################################

import byxtal.lattice as gbl
import byxtal.csl_utility_functions as cuf
import byxtal.find_csl_dsc as fcd
import numpy as np
import byxtal.tools as gbt
import byxtal.misorient_fz as mfz
import byxtal.disorient_symm_props as dsp

import os
import inspect
import byxtal
byxtal_dir = os.path.dirname((inspect.getfile(byxtal)))

## Directory and file names
pkl_dir = byxtal_dir+'/tests/pkl_files/'
##############################################################


l1 = gbl.Lattice()
sig_type = 'common'
l_p_po = l1.l_p_po

### For cubic only odd Sigma numbers exist!
n1 = 1
n2 = 5
sig_nums = 2*np.arange(n1,n2)+1

num_sigs = 0
sig_mats = {}
csl_mats = {}
dsc_mats = {}
csl_bp_props = {}

for sig_num in sig_nums:
    s1 = cuf.csl_rotations(sig_num, sig_type, l1)
    for ct1 in range(np.shape(s1['N'])[0]):
        bp_symm_grp_props = {}
        sig_id = str(sig_num)+'_'+str(ct1+1)
        print(sig_id)
        #### Store the sigma-misorientation (in 'p' reference frame)
        T_p1top2_p1 = s1['N'][ct1]/s1['D'][ct1]
        sig_mats[sig_id] = T_p1top2_p1

        l_csl_p, l_dsc_p = fcd.find_csl_dsc(l_p_po, T_p1top2_p1, 1e-6)
        csl_mats[sig_id] = l_csl_p; dsc_mats[sig_id] = l_dsc_p;

        #### Generate boundary-planpe orientations
        l_p_po = l1.l_p_po
        l_po_p = np.linalg.inv(l_p_po)
        T_p1top2_po1 = np.dot(l_p_po, np.dot(T_p1top2_p1, l_po_p))

        ## Find the corresponding disorientation
        quat1 = gbt.mat2quat(T_p1top2_po1)
        # print(quat1)
        dis_quat1 = mfz.misorient_fz(quat1, l1.cryst_ptgrp)
        # print(dis_quat1)
        x_g, y_g, z_g, bp_symm_grp = dsp.disorient_symm_props(dis_quat1, l1.cryst_ptgrp)
        symm_grp_ax = (np.vstack((x_g, y_g, z_g))).transpose()
        bp_symm_grp_props['symm_grp_ax'] = symm_grp_ax
        bp_symm_grp_props['bp_symm_grp'] = bp_symm_grp
        csl_bp_props[sig_id] = bp_symm_grp_props

import pickle as pkl;
pkl_name = pkl_dir+l1.elem_type+'_csl_bp_props_common_rotations.pkl'
csl_props = {}
csl_props['sig_mats'] = sig_mats
csl_props['csl_mats'] = csl_mats
csl_props['dsc_mats'] = dsc_mats
csl_props['csl_bp_props'] = csl_bp_props

jar = open(pkl_name, 'wb')
pkl.dump(csl_props, jar)
jar.close()

