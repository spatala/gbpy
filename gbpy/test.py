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


path_name = './output/'
output_path = './results/data/'
lat_par = 4.05
non_p = 2
weight_1= 0.5
CohEng= -3.35999998818377  #  calculated from in.cohesive
str_alg = "csc"
csc_tol = .1

rCut = 2*lat_par
Tm = 1000
weight_1 = .5
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
SC_tol = 5 * lat_par
directory_m = './lammps_dump/output/gb_attr_Al_S7_1_N1_4_2_1_N2_-4_-1_-2/accepted/'
directory = [directory_m]
energy = []
iteration = []
j = os.listdir(directory_m)
for var in j:
    num = re.findall('.*[._]([0-9]+)', var)
    print(var)
    filename_0 = directory_m + var 
    data = uf.compute_ovito_data(filename_0)
    eng = uf.cal_GB_E(data, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)
    energy = energy + [eng]
    # print(eng)
    iteration = iteration + [int(num[0])]

name = './results/heat.txt'
np.savetxt(name, np.column_stack([iteration,energy]))