import pickle as pkl;
import numpy as np;

import os;
import byxtal
path = os.path.dirname(byxtal.__file__)+'/data_files/'

fstr = ['mats', 'quats'];

prop_grps = ['C1', 'C2', 'C3', 'C4', 'C6', 'D2', 'D3', 'D4', 'D6', 'D8', 'T', 'O']
laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Oh']
# laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Th', 'Oh']
noncentro_grps = ['Cs']
# noncentro_grps = ['Cs', 'S4', 'S6', 'C2v', 'C3v', 'C4v', 'C6v', 'D2d', 'D3h', 'Td']


ptgrps = prop_grps+laue_grps+noncentro_grps;

for fstr1 in fstr:
	for cgrp in ptgrps:
		pkl_file = path+'symm_' + fstr1 + '_' + cgrp + '.pkl'
		jar=open(pkl_file, 'rb');
		if fstr1 == 'mats':
			symm_mats = pkl.load(jar);
			# print(symm_mats);
		if fstr1 == 'quats':
			symm_quats = pkl.load(jar);
			# print(symm_quats);
		jar.close();


# #!/bin/bash

# import numpy as np;
# import byxtal.generate_symm_ops as gso;
# import byxtal.tools as gbt;


# prop_grps = ['C1', 'C2', 'C3', 'C4', 'C6', 'D2', 'D3', 'D4', 'D6', 'D8', 'T', 'O']
# laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Oh']
# # laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Th', 'Oh']
# noncentro_grps = ['Cs']
# # noncentro_grps = ['Cs', 'S4', 'S6', 'C2v', 'C3v', 'C4v', 'C6v', 'D2d', 'D3h', 'Td']


# ptgrps = prop_grps+laue_grps+noncentro_grps;

# op_type = 'matrices';
# for cgrp in ptgrps:
# 	print(cgrp)
# 	gso.save_symm_pkl(cgrp, op_type);

# # axang = np.array([0,0,1,0,1])
# # q1 = gbt.axang2quat(axang);

# # cryst_ptgrp = 'Oh';
# # symm_quats = gso.generate_symm_quats(cryst_ptgrp, tol=1e-10);
# # gso.save_symm_pkl(cryst_ptgrp, 'quats')
