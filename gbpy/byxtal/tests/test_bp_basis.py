import numpy as np
import numpy.linalg as nla
import gbpy.byxtal.bp_basis as bpb
import gbpy.byxtal.lattice as gbl
import gbpy.byxtal.integer_manipulations as int_man
import gbpy.byxtal.reduce_po_lat as rpl
import pickle as pkl
import gbpy.byxtal.find_csl_dsc as fcd

import os
import inspect
import gbpy
byxtal_dir = os.path.dirname((inspect.getfile(gbpy.byxtal)))

## Directory and file names
pkl_dir = byxtal_dir+'/tests/pkl_files/'
##############################################################

l1 = gbl.Lattice()
pkl_name = pkl_dir+l1.elem_type+'_csl_common_rotations.pkl'
jar = open(pkl_name, "rb" )
lat_sig_attr = pkl.load(jar)
jar.close()

sig_rots = lat_sig_attr['sig_rots']
l_p_po = lat_sig_attr['l_p_po']
sig_vars = lat_sig_attr['sig_var']

sig_num = 3
ct1 = 0

s1 = sig_rots[str(sig_num)]
Nmats = s1['N']
Dmats = s1['D']
sz1 = np.shape(Nmats)[0]

Nmat = Nmats[ct1]
Dmat = Dmats[ct1]
T_p1top2_p1 = Nmat/Dmat
print(T_p1top2_p1)

t_mat = np.copy(T_p1top2_p1)
inds_type  = 'normal_go'
T_ref = 'g1'
# inds = np.array([2, 3, 1])
# inds = np.array([2, 1, 1])
inds = np.array([4, 2, 2])

l_2d_csl_p1, l_pl1_p1, l_pl2_p1 = bpb.gb_2d_csl(inds, t_mat, l_p_po, inds_type, T_ref)

# l_po_p = nla.inv(l_p_po)
# l_pR_po = fcd.reciprocal_mat(l_p_po)
# l_po_pR = nla.inv(l_pR_po)


# if inds_type == 'normal_go':
#     bp_po1 = np.array(inds)
# elif inds_type == 'miller_index':
#     bp_p1R = np.array(inds)
#     bp_po1 = l_pR_po.dot(bp_p1R)
# elif inds_type == 'normal_g':
#     bp_p1 = np.array(inds)
#     bp_po1 = l_p_po.dot(bp_p1)

# bp_p1R = l_po_pR.dot(bp_po1)
# miller1_ind, tm1 = int_man.int_approx(bp_p1R, 1e-6)
# l_pl1_p1 = bpb.bp_basis(miller1_ind)
# l_sig2_sig1 = rpl.reduce_po_lat(l_pl1_p1, l_p_po, 1e-6)
# l_pl1_p1 = l_pl1_p1.dot(l_sig2_sig1)


# l_p2_p1 = np.copy(t_mat)
# l_po2_po1 = l_p_po.dot(l_p2_p1.dot(l_po_p))
# l_po1_po2 = nla.inv(l_po2_po1)

# bp_po2 = -l_po1_po2.dot(bp_po1)
# bp_p2R = l_po_pR.dot(bp_po2)
# miller2_ind, tm1 = int_man.int_approx(bp_p2R, 1e-6)
# l_pl2_p2 = bpb.bp_basis(miller2_ind)
# l_sig2_sig1 = rpl.reduce_po_lat(l_pl2_p2, l_p_po, 1e-6)
# l_pl2_p2 = l_pl2_p2.dot(l_sig2_sig1)

# l_pl2_p1 = l_p2_p1.dot(l_pl2_p2)



# l_csl_p1 = fcd.csl_finder(t_mat, l_p_po, 1e-6)
# l_csl_po1 = l_p_po.dot(l_csl_p1)
# l_cslR_po1 = fcd.reciprocal_mat(l_csl_po1)
# l_po1_cslR = nla.inv(l_cslR_po1)

# n_cslR = l_po1_cslR.dot(bp_po1)
# mind_cslR, tm1 = int_man.int_approx(n_cslR, 1e-6)

# l_bpb_csl = bpb.bp_basis(mind_cslR)

# l_bpb_p1 = l_csl_p1.dot(l_bpb_csl)
# l_sig2_sig1 = rpl.reduce_po_lat(l_bpb_p1, l_p_po, 1e-6)
# l_2d_csl_p1 = l_bpb_p1.dot(l_sig2_sig1)



# bpb.check_2d_csl(l_pl1_p1, l_pl2_p1, l_2d_csl_p1)

# m_ind = [1,2,3]

# bpb1 = bpb.bp_basis(m_ind)



# sig_num = 19

# l1 = gbl.Lattice('cP_Id')
# pkl_name = pkl_dir+l1.elem_type+'_csl_common_rotations.pkl'
# jar = open(pkl_name, "rb" )
# lat_sig_attr = pkl.load(jar)
# jar.close()


# print(sig_num)
# s1 = sig_rots[str(sig_num)]
# Nmats = s1['N']
# Dmats = s1['D']
# sz1 = np.shape(Nmats)[0]
# csl_p_mats = {}

# # for ct1 in range(sz1):
# #     print(ct1)
# ct1 = 1
# Nmat = Nmats[ct1]
# Dmat = Dmats[ct1]
# T_p1top2_p1 = Nmat/Dmat
# print(T_p1top2_p1)

# bpb2 = bpb.gb_2d_csl(m_ind, T_p1top2_p1, l_p_po, 'miller_index', 'g1')

# ### LLL Reduction
# array([[ 0.,  1.],
#        [-3.,  1.],
#        [ 2., -1.]])
