import byxtal.tools as gbt
import byxtal.lattice as gbl
import pickle as pkl
import numpy as np
import numpy.linalg as nla
import byxtal.misorient_fz as mfz
import gbpy.generate_hkl_indices as ghi

########################################################################
pkl_name = 'cF_Id_byxtal_props.pkl';
jar = open(pkl_name, 'rb');

csl_props = pkl.load(jar)
sig_mats = csl_props['sig_mats']
csl_mats = csl_props['csl_mats']
dsc_mats = csl_props['dsc_mats']
bxt_symm_props = csl_props['csl_symm_props']
jar.close()
########################################################################

########################################################################
pkl_name = 'bp_list_fcc.pkl';
jar = open(pkl_name, 'rb');

hkl_sig_inds = pkl.load(jar)
jar.close()
########################################################################

s1_keys = list(sig_mats.keys())

# ind1 = np.random.randint(1, len(s1_keys))
ind1 = 638
sig_id = s1_keys[ind1]

# l_CSLbpb_CSLp = {}
# l_CSLbpbSig_CSLp = {}

# lat_par = l1.lat_params['a']
# rCut = lat_par*3
# A_cut = (rCut+lat_par)**2


# l_CSLbpb_CSLp[sig_id] = l_CSLbpb_CSLp_mat
# l_CSLbpbSig_CSLp_mat = ghi.gen_Acut_bpb(l_CSLbpb_CSLp_mat, l_csl_po, rCut, A_cut)
# l_CSLbpbSig_CSLp[sig_id] = l_CSLbpbSig_CSLp_mat

# gb_props['l_CSLbpb_CSLp'] = l_CSLbpb_CSLp
# gb_props['l_CSLbpbSig_CSLp'] = l_CSLbpbSig_CSLp

