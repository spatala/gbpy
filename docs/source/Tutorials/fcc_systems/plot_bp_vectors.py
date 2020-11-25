import pickle as pkl
import byxtal.lattice as gbl
import byxtal.find_csl_dsc as fcd
import numpy as np
import gbpy.generate_hkl_indices as ghi
import numpy.linalg as nla

def uvecs_symm(nu_vec, symm_mats):
    nsz = np.shape(symm_mats)[0]
    nsz1 = np.shape(nu_vec)[0]

    nuvecs_symm = np.zeros((nsz1*nsz, 3))
    for ct1 in range(nsz):
        i1 = ct1*nsz1
        i2 = (ct1+1)*nsz1
        tvec1 = (symm_mats[ct1].dot(nu_vec.transpose())).transpose()
        nuvecs_symm[i1:i2,:] = tvec1

    ### Return vectors with +z coordinate
    tind1 = np.where(nuvecs_symm[:,2] > 0)[0]

    return nuvecs_symm[tind1, :]


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

l_p_po = hkl_sig_inds['l_p_po']

s1_keys = list(hkl_sig_inds.keys())

# ind1 = np.random.randint(1, len(s1_keys))
# ind1 = 638
ind1 = 907
sig_id = s1_keys[ind1]

l_csl_p = csl_mats[sig_id]
l_csl_po = l_p_po.dot(l_csl_p)

symm_prop = bxt_symm_props[sig_id]
symm_grp_ax = symm_prop['symm_grp_ax']
bxt_symm_grp = symm_prop['bxt_symm_grp']

l_csl_props = {}
l_csl_props['l_csl_po'] = l_csl_po
l_csl_props['symm_grp_ax'] = symm_grp_ax
l_csl_props['bp_symm_grp'] = bxt_symm_grp
num1 = 3
# hkl_inds = ghi.gen_hkl_props(l_csl_props, num1)
hkl_inds = ghi.gen_hkl_inds(num1)
import byxtal.integer_manipulations as iman

nsz = np.shape(hkl_inds)[0]
hkl1_inds = 0*hkl_inds
for ct1 in range(nsz):
    t1 = hkl_inds[ct1]
    hkl1_inds[ct1,:], tm1 = iman.int_approx(t1, 1e-6)


l_p_po = l_csl_props['l_csl_po']
l_rp_po = fcd.reciprocal_mat(l_p_po)
l_po_rp = nla.inv(l_rp_po)
hkl_po = (np.dot(l_rp_po, hkl_inds.transpose())).transpose()

nu_po = hkl_po/np.tile(np.sqrt(np.sum(hkl_po**2, axis=1)), (3,1)).transpose()

hkl1_rp = (np.dot(l_po_rp, nu_po.transpose())).transpose()

nsz = np.shape(hkl_inds)[0]
hkl2_rp = 0*hkl_inds
for ct1 in range(nsz):
    t1 = hkl1_rp[ct1]
    hkl2_rp[ct1,:], tm1 = iman.int_approx(t1, 1e-6)

# mInd_arr = hkl_sig_inds[sig_id]
# n_uvecs = ghi.conv_hkl_uvecs(mInd_arr, l_csl_po)


# symm_grp_ax1 = symm_grp_ax.transpose()
# nu_vec1 = (symm_grp_ax1.dot(n_uvecs.transpose())).transpose()



# import os
# import byxtal
# # path = os.path.dirname(byxtal.__file__)+'/tests/';
# path = os.path.dirname(byxtal.__file__)+'/data_files/'
# symm_pkl = path+'symm_mats_'+bxt_symm_grp+'.pkl'

# # csl_pkl = path+l1.pearson+'_Id_csl_common_rotations.pkl'
# jar = open(symm_pkl,'rb')
# symm_mats = pkl.load(jar)
# jar.close()

# nu_vec  = uvecs_symm(nu_vec1, symm_mats)



# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(10,10))

# xs = nu_vec1[:,0]
# ys = nu_vec1[:,1]
# zs = nu_vec1[:,2]
# x1 = xs/(1+zs)
# y1 = ys/(1+zs)
# plt.scatter(x1, y1, marker='o')


# xs = nu_vec [:,0]
# ys = nu_vec [:,1]
# zs = nu_vec [:,2]
# x1 = xs/(1+zs)
# y1 = ys/(1+zs)
# plt.scatter(x1, y1, marker='*', alpha=0.5)

# ## Plot circle
# th1 = np.linspace(0, 2*np.pi, 100)
# x1 = np.cos(th1)
# y1 = np.sin(th1)
# plt.plot(x1, y1)

# ## Plot X-axis
# x1 = [-1.2,1.2]
# y1 = [0, 0]
# plt.plot(x1, y1)

# ## Plot X-axis
# y1 = [-1.2,1.2]
# x1 = [0, 0]
# plt.plot(x1, y1)

# plt.axis('equal')
# plt.axis('off')

# plt.show()
