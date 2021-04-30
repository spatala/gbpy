import byxtal.lattice as gbl;
import byxtal.csl_utility_functions as cuf;
import numpy as np;
from sympy.matrices import Matrix, eye, zeros;

########################################################
#### Input Parameters
fsave = False
num = 10
########################################################

# l1 = gbl.Lattice('cP_Id')
l1 = gbl.Lattice()
# l1 = gbl.Lattice('hP_Id')

sig_type = 'common'
l_p_po = l1.l_p_po

lat_sig_attr = {}
sig_rots = {}

lat_sig_attr['l_p_po'] = l_p_po;

sig_var = 2*np.arange(0,num)+1
sig_vars = []
for sig_num in sig_var:
    print(sig_num)
    s1 = cuf.csl_rotations(sig_num, sig_type, l1);
    if np.size(s1['N']) > 0:
        sig_vars.append(sig_num)
        sig_rots[str(sig_num)] = s1


lat_sig_attr['sig_rots'] = sig_rots
lat_sig_attr['l_p_po'] = l_p_po
lat_sig_attr['sig_var'] = sig_vars

if fsave:
    import pickle as pkl
    pkl_name = 'pkl_files/'+l1.elem_type+'_csl_common_rotations.pkl'
    jar = open(pkl_name, "wb" )
    pkl.dump(lat_sig_attr, jar, protocol=2)
    jar.close()

