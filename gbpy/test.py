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
# directory_a = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/test_n1/arash/'
# directory_m = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/test_n1/leila/'
# path = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/'
path_name = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/result/'
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
directory_a = '/home/leila/Downloads/15_al_S5_0_N1_1_-2_1_N2_-1_1_-2/accepted_steps/'
directory_m = '/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/test/accepted/'
directory = [directory_m, directory_a]
directory = [directory_m]
# directory = ['/home/leila/Leila_sndhard/codes/gbmc_python/gbmc_v0/gbmc_v0/gbmc_v0/lammps_dump/test/accepted/']
for n_run in range(1,2):
    # directory_a = path + 'test_n' + str(n_run) + '/arash/'
    # directory_m = path + 'test_n' + str(n_run) + '/leila/'
    j = 0
    for dir in directory:
        a = os.listdir(dir)
        energy = []
        iteration = []
        for i in a:
            num = re.findall('.*[._]([0-9]+)', i)
            filename_0 = dir + i
            data = uf.compute_ovito_data(filename_0)
            eng = uf.cal_GB_E(data, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)
            energy = energy + [eng]
            # print(eng)
            iteration = iteration + [int(num[0])]
        
        if j == 1:
            name = path_name + 'arash/' + str(n_run) + '.txt'
        else:
            name = path_name + 'leila/' + str(n_run) + '.txt'
        j += 1 
        np.savetxt(name, np.column_stack([iteration,energy]))
    # import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# axes.plot(iteration, energy, 'o-', markeredgewidth=0)
# plt.show()

