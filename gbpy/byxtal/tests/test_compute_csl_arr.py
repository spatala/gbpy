import byxtal.lattice as gbl
import byxtal.integer_manipulations as iman
import byxtal.find_csl_dsc as fcd
import numpy as np
import byxtal.compute_csl as cCSL
import byxtal.reduce_po_lat as rpl

import numpy.linalg as nla
import pickle as pkl

import pathlib
currDir = pathlib.Path().absolute()
print(currDir)

## Directory and file names
pkl_dir = currDir.joinpath('pkl_files/')
pkl_inp_fname = 'csl_inp_mats.pkl'
pkl_out_fname = 'csl_out_mats.pkl'
##############################################################


##############################################################
l1 = gbl.Lattice()
pkl_name = pkl_dir.joinpath(l1.elem_type+'_csl_common_rotations.pkl')
jar = open(pkl_name, "rb" )
lat_sig_attr = pkl.load(jar)
jar.close()

sig_rots = lat_sig_attr['sig_rots']
l_p_po = lat_sig_attr['l_p_po']
sig_var = lat_sig_attr['sig_var']

n_mats = 0
for sig_num in sig_var:
    s1 = sig_rots[str(sig_num)]
    Nmats = s1['N']
    sz1 = np.shape(Nmats)[0]
    for ct1 in range(sz1):
        n_mats = n_mats + 1

print(n_mats)
csl_N_mats = np.zeros((n_mats, 3, 3))
csl_D_mats = np.zeros((n_mats, 3, 3))

tct1 = 0
for sig_num in sig_var:
    s1 = sig_rots[str(sig_num)]
    Nmats = s1['N']
    Dmats = s1['D']
    sz1 = np.shape(Nmats)[0]
    for ct1 in range(sz1):
        Nmat = Nmats[ct1]
        Dmat = Dmats[ct1]
        csl_N_mats[tct1] = Nmat
        csl_D_mats[tct1] = Dmat
        tct1 = tct1 + 1

csl_inp_mats = {}
csl_inp_mats['N'] = csl_N_mats
csl_inp_mats['D'] = csl_D_mats
csl_inp_mats['l_p_po'] = l_p_po
csl_inp_mats['tol'] = 0.01

pkl_name = pkl_dir.joinpath(pkl_inp_fname)
jar = open(pkl_name, 'wb')
pkl.dump(csl_inp_mats, jar, protocol=2)
jar.close()
##############################################################

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
#    print(sig_num)
   l_csl1_p = cCSL.compute_csl_grimmer(Nmat, sig_num, Sz[0])
   l_csl2_csl1 = rpl.reduce_po_lat(l_csl1_p, l_p_po, tol1)

#    print(l_csl2_csl1)
#    print('================')



