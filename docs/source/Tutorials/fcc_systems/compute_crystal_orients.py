import pickle as pkl;
import byxtal.lattice as gbl;
import byxtal.find_csl_dsc as fcd;
import byxtal.integer_manipulations as iman;
import byxtal.bp_basis as bpb;
import numpy as np;
import math as mt;
import gbpy.util_funcs_create_byxtal as uf
import numpy.linalg as nla


##############################################################################
########## Variable Definitions
## l_Cbpb_Cp:
## Basis of the 2D lattice in the CSLp reference frame for the
## given (hkl) plane.

## l_CRbpb_Cp:
## Basis of the 2D lattice such that the basis vectors have
## length that is greater than rCut and the basis vectors are similar
## in lengths and the angle between them is close to 90.
## Defined in the CSLp reference frame
##############################################################################


##############################################################################
l1 = gbl.Lattice('Al')
l_p_po = l1.l_p_po
l_po_p = nla.inv(l_p_po)

##############################################################################
##### Load bicrystallographic properties
pkl_name = 'cubic_gb_props.pkl'
jar = open(pkl_name, 'rb')
gb_props = pkl.load(jar)
jar.close()
lSig_CSLbpb_CSLp = gb_props['lSig_CSLbpb_CSLp']
lSig_CSLbpb_Rcut_CSLp = gb_props['lSig_CSLbpb_Rcut_CSLp']
hkl_sig_inds = gb_props['hkl_sig_inds']


csl_pkl = l1.pearson+'_Id_byxtal_props.pkl'
jar = open(csl_pkl,'rb')
s1=pkl.load(jar)
jar.close()

sig_mats = s1['sig_mats']
csl_mats = s1['csl_mats']
dsc_mats = s1['dsc_mats']
csl_bp_props = s1['csl_bp_props']

s1_keys = list(sig_mats.keys())

sig_id = s1_keys[0]

l_csl_p1 = csl_mats[sig_id]
l_csl_po1 = l_p_po.dot(l_csl_p1)

hkl_inds = hkl_sig_inds[sig_id]

tct1 = 2

l_Cbpb_Cp = lSig_CSLbpb_CSLp[sig_id][tct1]
l_CRbpb_Cp = lSig_CSLbpb_Rcut_CSLp[sig_id][tct1]
hkl1 = hkl_inds[tct1]

l_bp_po1 = l_csl_po1.dot(l_CRbpb_Cp)

symm_grp_ax = csl_bp_props[sig_id]['symm_grp_ax']
bp_symm_grp = csl_bp_props[sig_id]['bp_symm_grp']

l_p2_p1 = sig_mats[sig_id];
gb_ID = uf.get_gb_uID(l_bp_po1, l_p2_p1, l_p_po, bp_symm_grp, symm_grp_ax, sig_id);
print(gb_ID);




