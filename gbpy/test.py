import util_funcs as uf
import pad_dump_file as pdf
import vv_props as vvp
import lammps_dump_writer as ldw
import lammps_script_writer as lsw
import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm
from shutil import copyfile
import numpy as np
import os
import re
import byxtal.lattice as gbl

path_name = './output/'
output_path = './results/data/'
l1 = gbl.Lattice('Mg_sun')
non_p = 2
weight_1= 0.5
CohEng= -1.5287  #  calculated from in.cohesive
str_alg = "ptm"
csc_tol = .1

rCut = 2*5 * l1.lat_params['a']
Tm = 923.1
weight_1 = .5
tol_fix_reg = 5 * 5 * l1.lat_params['a']  # the width of rigid traslation region
SC_tol = 5 * 5 * l1.lat_params['a']
directory_m = './lammps_dump/output/gb_attr_Mg_sun_S13_1_N1_0_0_1_N2_0_0_-1/accepted/'
directory = [directory_m]
energy = []
iteration = []
j = os.listdir(directory_m)
for var in j:
    num = re.findall('.*[._]([0-9]+)', var)
    print(var)
    filename_0 = directory_m + var 
    data = uf.compute_ovito_data(filename_0)
    eng = uf.cal_GB_E(data, weight_1, non_p, l1, CohEng, str_alg, csc_tol)
    energy = energy + [eng]
    print(eng)
    iteration = iteration + [int(num[0])]

name = './results/heat.txt'
np.savetxt(name, np.column_stack([iteration,energy]))