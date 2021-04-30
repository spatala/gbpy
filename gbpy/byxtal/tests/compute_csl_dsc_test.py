import pickle as pkl
import byxtal.lattice as gbl;
import numpy as np;
import byxtal.find_csl_dsc as fcd

import os
import inspect
import byxtal
byxtal_dir = os.path.dirname((inspect.getfile(byxtal)))

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

sig_num = 19
print(sig_num)
s1 = sig_rots[str(sig_num)]
Nmats = s1['N']
Dmats = s1['D']
sz1 = np.shape(Nmats)[0]
csl_p_mats = {}

for ct1 in range(sz1):
    print(ct1)
    Nmat = Nmats[ct1]
    Dmat = Dmats[ct1]

    Tmat = Nmat/Dmat
    print('++++++++++++++++++++++++++++++++++++')
    l_csl3_p, l_dsc3_p = fcd.find_csl_dsc(l_p_po, Tmat)
    print('++++++++++++++++++++++++++++++++++++')

