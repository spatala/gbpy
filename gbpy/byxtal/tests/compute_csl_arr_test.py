import gbpy.byxtal.lattice as gbl
import gbpy.byxtal.integer_manipulations as iman
import gbpy.byxtal.find_csl_dsc as fcd
import numpy as np
import subprocess
import numpy.linalg as nla

import os
import inspect
import gbpy
byxtal_dir = os.path.dirname((inspect.getfile(gbpy.byxtal)))
import pickle as pkl

## Directory and file names
pkl_dir = byxtal_dir+'/tests/pkl_files/'
pkl_inp_fname = 'csl_inp_mats.pkl'
pkl_out_fname = 'csl_out_mats.pkl'
##############################################################


##############################################################
l1 = gbl.Lattice()
pkl_name = pkl_dir+l1.elem_type+'_csl_common_rotations.pkl'
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

pkl_name = pkl_dir+pkl_inp_fname
jar = open(pkl_name, 'wb')
pkl.dump(csl_inp_mats, jar, protocol=2)
jar.close()
##############################################################

exec_str = byxtal_dir+'/compute_csl_arr.py'
run_lst = []
run_lst.append(exec_str)

run_lst.append(pkl_dir)
run_lst.append(pkl_inp_fname)
run_lst.append(pkl_out_fname)

result = subprocess.run(run_lst, stdout=subprocess.PIPE)

pkl_name = pkl_dir+pkl_out_fname
jar = open(pkl_name, 'rb')
l_csl_p_mats = pkl.load(jar, encoding='latin1')
jar.close()

nsz = np.shape(l_csl_p_mats)[0]

csl_p_mats = {}
for ct1 in range(nsz):
    print(ct1)
    Nmat = csl_N_mats[ct1]
    Dmat = csl_D_mats[ct1]
    sig_num = int(np.unique(Dmat)[0])

    l_csl2_p = l_csl_p_mats[ct1]
    T_p1top2_p1 = Nmat/Dmat
    check_val = fcd.check_csl(l_csl2_p, l_p_po, T_p1top2_p1, sig_num, True)
    if not(check_val):
        raise Exception("CSL Finder failed")
    else:
        csl_p_mats[str(ct1+1)] = np.array(l_csl2_p, dtype='int64')
        print('++++++++++++++++++')


