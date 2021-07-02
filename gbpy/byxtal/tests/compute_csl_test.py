##############################################################################
### Test function to compute the CSL lattice for cubic crystals
##############################################################################
import numpy as np
import pickle as pkl
import numpy.linalg as nla
import gbpy.byxtal.lattice as gbl
import gbpy.byxtal.integer_manipulations as iman
import gbpy.byxtal.find_csl_dsc as fcd

import os
import inspect
# import byxtal
import gbpy
byxtal_dir = os.path.dirname((inspect.getfile(gbpy.byxtal)))

## Directory and file names
pkl_dir = byxtal_dir+'/tests/pkl_files/'
##############################################################

sig_num = 19

l1 = gbl.Lattice()
pkl_name = pkl_dir+l1.elem_type+'_csl_common_rotations.pkl'
jar = open(pkl_name, "rb" )
lat_sig_attr = pkl.load(jar)
jar.close()

sig_rots = lat_sig_attr['sig_rots']
l_p_po = lat_sig_attr['l_p_po']
sig_vars = lat_sig_attr['sig_var']

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
    T_p1top2_p1 = Nmat/Dmat
    tol1 = 1e-6
    Sigma = fcd.sigma_calc(T_p1top2_p1, tol1)
    ########################################################################
    l_csl_p = fcd.csl_finder(T_p1top2_p1, l_p_po, tol1)
    print_check = True
    check_val1 = fcd.check_csl(l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)
    ########################################################################

    ########################################################################
    l_dsc_p = fcd.dsc_finder(T_p1top2_p1, l_p_po, tol1)
    check_val2 = fcd.check_dsc(l_dsc_p, l_csl_p, l_p_po, T_p1top2_p1, Sigma, print_check)
    ########################################################################

    ########################################################################
    print([check_val1, check_val2])
    if (not(check_val1 and check_val2)):
        raise Exception("Error in Computing CSL or DSC Lattices.")
    ########################################################################


