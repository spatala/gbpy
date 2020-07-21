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

#  --------------------------------------------------------------------------
#  Define the input
#  --------------------------------------------------------------------------
lat_par = 4.05
rCut = 2*lat_par
CohEng= -3.35999998818377  #  calculated from in.cohesive
Tm = 933.5
weight_1 = .5
tol_fix_reg = 5 * lat_par  # the width of rigid traslation region
SC_tol = 5 * lat_par
# str_alg = "ptm"
str_alg = "csc"
csc_tol = .1
# method = "anneal"
method = "min"
Etol_val=1e-25
Ftol_val=1e-25
if method=="anneal":
    Etol_val0=1e-5
    Ftol_val0=1e-5
else:
    Etol_val0=1e-25
    Ftol_val0=1e-25

MaxIter_val=5000
MaxEval_val=10000
Iter_heat_val=1000
Iter_equil_val=3000
Iter_cool_val=1000
#  --------------------------------------------------------------------------
#  Define the path to dump files
#  --------------------------------------------------------------------------
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
pot_path = './lammps_dump/'  # the path for the potential
dump_path = './lammps_dump/test/'
pkl_file = './tests/data/gb_attr_Al_S3_1_N1_4_-1_-3_N2_-3_4_-1.pkl'
initial_dump = 'tests/data/dump_1'  # the name of the dump file that
output =  dump_path + 'dump_min'
#  --------------------------------------------------------------------------
#  Create lammps dump file for pkl file
#  --------------------------------------------------------------------------
box_bound, dump_lamp, box_type = ldw.lammps_box(lat_par, pkl_file) # lammps creates from the pkl file
ldw.write_lammps_dump(initial_dump, box_bound, dump_lamp, box_type)  # writing the dump file

#  --------------------------------------------------------------------------
#  Define the path to dump files
#  --------------------------------------------------------------------------
out_min_1 = dump_path + 'dump.0' # the output of previous step
fil_name = 'in.min_1'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
lsw.run_lammps_min(initial_dump, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_1,\
               step=1, Etol=Etol_val0, Ftol=Ftol_val0, MaxIter=MaxIter_val, MaxEval=MaxEval_val)
if method == "anneal":
    fil_name1 = 'in.anneal'
    out_heat = dump_path + 'heat.0'
    lsw.run_lammps_anneal(out_min_1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
                    Tm,  step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val, Iter_cool=Iter_cool_val)

    lines = open(out_heat, 'r').readlines()
    lines[1] = '0\n'
    out = open(out_heat, 'w')
    out.writelines(lines)
    out.close()

    fil_name2 = 'in.min_2'
    out_min_2 = dump_path + 'dump.1'
    lsw.run_lammps_min(out_heat, fil_name2, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_2,\
                step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val)

    filename_0 = out_min_2
else:
    filename_0 = out_min_1

#  --------------------------------------------------------------------------
#  Start MC
#  --------------------------------------------------------------------------
iter = 5000
ff = open('output', 'w')
for i in range(1, iter, 1):
    print(i)
    
    #  read the data
    data_0 = uf.compute_ovito_data(filename_0)
    non_p = uf.identify_pbc(data_0)
    #  find the gb atoms
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data_0, lat_par, non_p, str_alg, csc_tol)

    #  decide between remove and insertion
    choice = uf.choos_rem_ins()
    
    #  --------------------------------------------------------------------------
    #  If the choice is removal
    #  --------------------------------------------------------------------------
    if choice == "removal":
        p_rm = uf.RemProb(data_0, CohEng, GbIndex)
        ID2change = uf.RemIns_decision(p_rm)
        ff.write(filename_0 + '\n')

        ff.write(str(i) + ' ' + choice + ' ' + str(GbIndex[ID2change]) )
        ff.write('\n')
        print(GbIndex[ID2change])

        var2change = data_0.particles['Particle Identifier'][GbIndex[ID2change]]
        uf.atom_removal(filename_0, dump_path , GbIndex[ID2change], var2change)
        
        fil_name = 'in.min_1'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
        filename_rem = dump_path + 'rem_dump'
        # copyfile(filename_rem, dump_path + 'rem/rem_dump_' + str(i))
        filename_1 = dump_path + 'dump.' + str(i)
        #  --------------------------------------------------------------------------------------------------
        lsw.run_lammps_min(filename_rem, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, filename_1,\
               step=2, Etol=Etol_val0, Ftol=Ftol_val0, MaxIter=MaxIter_val, MaxEval=MaxEval_val)
        copyfile(filename_rem, dump_path + 'rem/min1_' + str(i))

        if method == "anneal":
            fil_name1 = 'in.anneal'
            out_heat = dump_path + 'heat.0'
            lsw.run_lammps_anneal(filename_1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
                                Tm,  step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val, Iter_cool=Iter_cool_val)

            lines = open(out_heat, 'r').readlines()
            lines[1] = '0\n'
            out = open(out_heat, 'w')
            out.writelines(lines)
            out.close()
            copyfile(out_heat, dump_path + 'rem/heat_' + str(i))


            fil_name2 = 'in.min_2'
            out_min_2 = dump_path + 'dump.' + str(i)
            lsw.run_lammps_min(out_heat, fil_name2, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_2,\
                            step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val)
            filename_1 = out_min_2

        #  --------------------------------------------------------------------------------------------------
        
        data_1 = uf.compute_ovito_data(filename_1)
        SC_boolean = uf.check_SC_reg(data_1, lat_par, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol)

        if str_alg == "ptm":
            assert data_0.particles['Structure Type'][GbIndex[ID2change]] !=1
        else:
            assert data_0.particles['c_csym'][GbIndex[ID2change]] > .1
        assert SC_boolean == [True, True]
    
        E_1 = uf.cal_GB_E(data_1, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)  #  after removal
        E_0 = uf.cal_GB_E(data_0, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)
        dE = E_1 - E_0
        if dE < 0:
            decision = "accept"
            print("finally accepted in removal")
        else:
            area = uf.cal_area(data_1, non_p)
            p_boltz = uf.p_boltz_func(dE, area, Tm)
            decision = uf.decide(p_boltz)

        if decision == "accept":
            print("accepted in botlzman removal")
            print(GbIndex[ID2change])
            copyfile(filename_1, dump_path + 'accepted/dump.' + str(i))
            filename_0 = filename_1

    #  --------------------------------------------------------------------------
    #  If the choice is insertion
    #  -------------------------------------------------------------------------- 
    else:
        ff.write(filename_0 + '\n' )
        pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data_0, lat_par, rCut, non_p, str_alg, csc_tol)
        tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
        cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, lat_par)
        cc_coors1 = vvp.wrap_cc(data_0.cell, cc_coors)
        Prob = uf.radi_normaliz(cc_rad)
        ID2change = uf.RemIns_decision(Prob)
        pos_add_atom = cc_coors[ID2change]
        atom_id = np.max(data_0.particles['Particle Identifier']+1)
        uf.atom_insertion(filename_0, dump_path, pos_add_atom, atom_id)
        ff.write(str(i) + ' ' + choice + ' ' + str(pos_add_atom) )
        ff.write('\n')

        filename_1 = dump_path + 'dump.' + str(i)
        #  -------------------------------------------------------------------------------------------------

        fil_name = 'in.min_1'  # the initila minimization lammps script write the in.min script and run it and create dump_minimized
        filename_ins = dump_path + 'ins_dump'
        # copyfile(filename_rem, dump_path + 'rem/rem_dump_' + str(i))
        filename_1 = dump_path + 'dump.' + str(i)
        #  --------------------------------------------------------------------------------------------------
        lsw.run_lammps_min(filename_ins, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, filename_1,\
               step=2, Etol=Etol_val0, Ftol=Ftol_val0, MaxIter=MaxIter_val, MaxEval=MaxEval_val)
        copyfile(filename_1, dump_path + 'ins/min1_' + str(i))
        if method == "anneal":
            fil_name1 = 'in.anneal'
            out_heat = dump_path + 'heat.0'
            lsw.run_lammps_anneal(filename_1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
                                Tm,  step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val, Iter_cool=Iter_cool_val)

            lines = open(out_heat, 'r').readlines()
            lines[1] = '0\n'
            out = open(out_heat, 'w')
            out.writelines(lines)
            out.close()
            copyfile(out_heat, dump_path + 'ins/heat_' + str(i))


            fil_name2 = 'in.min_2'
            out_min_2 = './lammps_dump/final'
            lsw.run_lammps_min(out_heat, fil_name2, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_2,\
                            step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val)
            filename_1 = out_min_2


        data_1 = uf.compute_ovito_data(filename_1)
        SC_boolean = uf.check_SC_reg(data_1, lat_par, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol)
        assert SC_boolean == [True, True]

        E_1 = uf.cal_GB_E(data_1, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)  #  after removal
        E_0 = uf.cal_GB_E(data_0, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol)
        dE = E_1 - E_0
        if dE < 0:
            decision = "accept"
        else:
            area = uf.cal_area(data_1, non_p)
            p_boltz = uf.p_boltz_func(dE, area, Tm)
            decision = uf.decide(p_boltz)

        if decision == "accept":
            copyfile(filename_1, dump_path + 'accepted/dump.' + str(i))
            filename_0 = filename_1

ff.close()







