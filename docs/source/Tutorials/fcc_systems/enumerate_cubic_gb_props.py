import byxtal.tools as gbt;
import byxtal.lattice as gbl
import pickle as pkl
import numpy as np
import numpy.linalg as nla
import byxtal.misorient_fz as mfz
import gbpy.generate_hkl_indices as ghi

l1=gbl.Lattice()
csl_pkl = l1.pearson+'_Id_byxtal_props.pkl'
l_p_po = l1.l_p_po
jar = open(csl_pkl,'rb')
s1=pkl.load(jar)
jar.close()


sig_mats = s1['sig_mats']
csl_mats = s1['csl_mats']
dsc_mats = s1['dsc_mats']
csl_bp_props = s1['csl_bp_props']


######################################################################
###### Boundary-plane matrix properties with A_cut
#####################################################################
lat_par = l1.lat_params['a']
rCut = lat_par*3
A_cut = (rCut+lat_par)**2

num1 = 3

s1_keys = list(csl_bp_props.keys())
hkl_sig_inds = {}
lSig_CSLbpb_CSLp = {}
lSig_CSLbpb_Rcut_CSLp = {}

# for sig_id in s1_keys:
sig_id = s1_keys[0]

print(sig_id)
T_p1top2_p1 = sig_mats[sig_id];

l_p_po = l1.l_p_po
l_po_p = nla.inv(l_p_po)

T_p1top2_po1 = l_p_po.dot(T_p1top2_p1.dot(l_po_p))

## Find the corresponding disorientation
Tmat = np.copy(T_p1top2_po1)
quat1 = gbt.mat2quat(Tmat)
dis_quat1 = mfz.misorient_fz(quat1, l1.cryst_ptgrp)
bp_symm_grp = csl_bp_props[sig_id]['bp_symm_grp']
symm_grp_ax = csl_bp_props[sig_id]['symm_grp_ax']
l_csl_p = csl_mats[sig_id]


l_csl_po = l_p_po.dot(l_csl_p)
l_csl_props = {}

l_csl_props['l_csl_po'] = l_csl_po
l_csl_props['symm_grp_ax'] = symm_grp_ax
l_csl_props['bp_symm_grp'] = bp_symm_grp

hkl_inds, l_CSLbpb_CSLp = ghi.gen_hkl_props(l_csl_props, num1)
l_CSLbpbSig_CSLp = ghi.gen_Acut_bpb(l_CSLbpb_CSLp, l_csl_po, rCut, A_cut)

hkl_sig_inds[sig_id] = hkl_inds
lSig_CSLbpb_CSLp[sig_id] = l_CSLbpb_CSLp
lSig_CSLbpb_Rcut_CSLp[sig_id] = l_CSLbpbSig_CSLp

########################################################################
pkl_name = 'cubic_gb_props.pkl';
jar = open(pkl_name, 'wb');

gb_props = {};
gb_props['lSig_CSLbpb_CSLp'] = lSig_CSLbpb_CSLp;
gb_props['lSig_CSLbpb_Rcut_CSLp'] = lSig_CSLbpb_Rcut_CSLp;
gb_props['hkl_sig_inds'] = hkl_sig_inds;

pkl.dump(gb_props, jar); jar.close();
########################################################################
