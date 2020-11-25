##################################################################
## Code to enumerate the properties of the CSL/DSC lattices for
## Sigma misorientations
## The code currently works for cubic lattices.
##################################################################

import byxtal.tools as gbt
import byxtal.lattice as gbl
import pickle as pkl
import numpy as np
import numpy.linalg as nla
import byxtal.misorient_fz as mfz
import gbpy.generate_hkl_indices as ghi


######################################################################
###### Boundary-plane matrix properties with A_cut
#####################################################################
l1=gbl.Lattice()
pkl_name = l1.elem_type+'_byxtal_props.pkl'
jar = open(pkl_name,'rb')
s1=pkl.load(jar)
jar.close()

sig_mats = s1['sig_mats']
csl_mats = s1['csl_mats']
dsc_mats = s1['dsc_mats']
bxt_symm_props = s1['csl_symm_props']

l_p_po = l1.l_p_po

num1 = 3


s1_keys = list(bxt_symm_props.keys())
hkl_sig_inds = {}

# for sig_id in s1_keys:
sig_id = s1_keys[0]
print(sig_id)

l_p_po = l1.l_p_po
l_po_p = nla.inv(l_p_po)

bp_symm_grp = bxt_symm_props[sig_id]['bxt_symm_grp']
symm_grp_ax = bxt_symm_props[sig_id]['symm_grp_ax']
l_csl_p = csl_mats[sig_id]


l_csl_po = l_p_po.dot(l_csl_p)
l_csl_props = {}

l_csl_props['l_csl_po'] = l_csl_po
l_csl_props['symm_grp_ax'] = symm_grp_ax
l_csl_props['bp_symm_grp'] = bp_symm_grp

hkl_inds = ghi.gen_hkl_props(l_csl_props, num1)
hkl_sig_inds[sig_id] = hkl_inds
