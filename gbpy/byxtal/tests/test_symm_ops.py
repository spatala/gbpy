import pickle as pkl
import numpy as np

import os

import pathlib
import byxtal

package_file_path = pathlib.Path(byxtal.__file__)
path = package_file_path.parent.joinpath('data_files')

fstr = ['mats', 'quats']

prop_grps = ['C1', 'C2', 'C3', 'C4', 'C6', 'D2', 'D3', 'D4', 'D6', 'D8', 'T', 'O']
laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Oh']
# laue_grps = ['Ci', 'C2h', 'C3i', 'C4h', 'C6h', 'D2h', 'D3d', 'D4h', 'D6h', 'D8h', 'Th', 'Oh']
noncentro_grps = ['Cs']
# noncentro_grps = ['Cs', 'S4', 'S6', 'C2v', 'C3v', 'C4v', 'C6v', 'D2d', 'D3h', 'Td']


ptgrps = prop_grps+laue_grps+noncentro_grps

for fstr1 in fstr:
	for cgrp in ptgrps:
		pkl_file = path.joinpath('symm_' + fstr1 + '_' + cgrp + '.pkl')
		jar=open(pkl_file, 'rb')
		if fstr1 == 'mats':
			symm_mats = pkl.load(jar)
			# print(symm_mats)
		if fstr1 == 'quats':
			symm_quats = pkl.load(jar)
			# print(symm_quats)
		jar.close()

