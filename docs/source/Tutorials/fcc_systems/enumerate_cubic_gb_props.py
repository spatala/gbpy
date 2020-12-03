import byxtal.tools as gbt
import byxtal.lattice as gbl
import pickle as pkl
import numpy as np
import numpy.linalg as nla
import byxtal.misorient_fz as mfz
import byxtal.integer_manipulations as iman
import gbpy.generate_hkl_indices as ghi
import gbpy.util_funcs_create_byxtal as uf
import byxtal.find_csl_dsc as fcd


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

bp_list = pkl.load(jar)
l_p_po = bp_list['l_p_po']
hkl_sig_inds = bp_list['miller_inds']

jar.close()
########################################################################

s1_keys = list(hkl_sig_inds.keys())
ind1 = 0
sig_id = s1_keys[ind1]
hkl_inds = hkl_sig_inds[sig_id]
l_csl_p1 = csl_mats[sig_id]
l_csl_po1 = l_p_po.dot(l_csl_p1)
l_po1_csl = nla.inv(l_csl_po1)


import os
import inspect
import byxtal
byxtal_dir = os.path.dirname((inspect.getfile(byxtal)))

# l_CSLbpb_CSLp = {}
# l_CSLbpbSig_CSLp = {}

# l1 = gbl.Lattice('Al')
# lat_par = l1.lat_params['a']
# rCut = lat_par*3
# A_cut = (rCut+lat_par)**2

# tct1 = 0
# hkl1 = np.zeros((1,3), dtype='int64')
# hkl1[0,:] = hkl_inds[tct1,:]

# l_CSLbpb_CSLp_mat = ghi.compute_hkl_bpb(hkl1)
# l_CSLbpbSig_CSLp_mat = ghi.gen_Acut_bpb(l_CSLbpb_CSLp_mat, l_csl_po1, rCut, A_cut)
# l_CRbpb_Cp = l_CSLbpbSig_CSLp_mat[0]

# l_bp_po1 = l_csl_po1.dot(l_CRbpb_Cp)

symm_grp_ax = bxt_symm_props[sig_id]['symm_grp_ax']
bp_symm_grp = bxt_symm_props[sig_id]['bxt_symm_grp']

pkl_id = byxtal_dir+'/data_files/symm_mats_'+bp_symm_grp+'.pkl'
jar = open(pkl_id, 'rb')
symm_mats = pkl.load(jar)
jar.close()

l_s_po1 = np.copy(symm_grp_ax)
l_po1_s = nla.inv(l_s_po1)

l_rcsl_po1 = fcd.reciprocal_mat(l_csl_po1)
l_rcsl_csl = l_po1_csl.dot(l_rcsl_po1)
l_csl_rcsl = nla.inv(l_rcsl_csl)


# ct1 = 1
nsz = np.shape(symm_mats)[0]
Tmats_csl = np.zeros((nsz,3,3), dtype='int64')

for ct1 in range(nsz):
    T_s = symm_mats[ct1]
    T_po1 = l_s_po1.dot(T_s.dot(l_po1_s))
    T_csl = l_po1_csl.dot(T_po1.dot(l_csl_po1))
    TI_csl, tm1 = iman.int_approx(T_csl)
    if iman.check_int_mat(T_csl, 1e-6):
        T_csl = np.around(T_csl)
        T_csl = T_csl.astype(int)
        Tmats_csl[ct1] = T_csl

    Tmat1 = l_csl_rcsl.dot(Tmats_csl[ct1].dot(l_rcsl_csl))




# l_p2_p1 = sig_mats[sig_id]
# gb_ID = uf.get_gb_uID(l_bp_po1, l_p2_p1, l_p_po, bp_symm_grp, symm_grp_ax, sig_id)

# zCut = 25*l1.lat_params['a']

# threeD_upts, sim_cell2 = uf.create_half_cryst(l_csl_p1, l_CRbpb_Cp, l_p_po, 'upper', zCut)

# l_p1_p2 = nla.inv(l_p2_p1)
# l_csl_p2 = l_p1_p2.dot(l_csl_p1)
# threeD_lpts, sim_cell1 = uf.create_half_cryst(l_csl_p2, l_CRbpb_Cp, l_p_po, 'lower', zCut)

# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# import plotting_routines as plr
# ## %matplotlib inline

# fig1 = plt.figure()
# plr.plot_3d_pts_box(fig1, threeD_pts1, sim_cell[:,0:3], sim_orig)
# plt.show()

# gb_attr = {}
# gb_attr['cell'] = sim_cell1
# gb_attr['upts'] = threeD_upts
# gb_attr['lpts'] = threeD_lpts

# pkl_name = 'gb_attr_'+gb_ID+'.pkl'
# jar = open(pkl_name,'wb')
# pkl.dump(gb_attr, jar)
# jar.close()
# ########################################################################################

