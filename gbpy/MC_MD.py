import gbpy.util_funcs as uf
import gbpy.pad_dump_file as pdf
import gbpy.vv_props as vvp
import gbpy.lammps_dump_writer as ldw
import gbpy.lammps_script_writer as lsw
from shutil import copyfile
import numpy as np
import sys
import gbpy.byxtal.lattice as gbl
#  --------------------------------------------------------------------------
#  Define the input
#  --------------------------------------------------------------------------



def MC_MD(gb_name, element, CohEng, Tm, lammps_exe_path, str_alg="ptm"):
    """
    """
    l1 = gbl.Lattice(str(element))  # 
    rCut = 2*l1.lat_params['a']
    method = "anneal"
    weight_1 = .5
    tol_fix_reg = 5 * l1.lat_params['a']  # the width of rigid traslation region
    SC_tol = 5 * l1.lat_params['a']
    csc_tol = .1
    Etol_val = 1e-25
    Ftol_val = 1e-25
    if method == "anneal":
        Etol_val0 = 1e-5
        Ftol_val0 = 1e-5
    else:
        Etol_val0 = 1e-25
        Ftol_val0 = 1e-25

    MaxIter_val = 10000
    MaxEval_val = 10000
    Iter_heat_val = 1000
    Iter_equil_val = 3000
    Iter_cool_val = 1000
    # gb_name = sys.argv[1]

    elem_type = l1.pearson
    if elem_type == 'cP':
        st_id = 5
    if elem_type == 'cI':
        st_id = 3  
    if elem_type == 'cF':
        st_id = 1
    if elem_type == 'hP':
        st_id = 2

    print('=============================================================================')
    print(gb_name)
    print('=============================================================================')
    #  --------------------------------------------------------------------------
    #  Define the path to dump files
    #  --------------------------------------------------------------------------
    # For my local pc##############################################################
    main_path = './lammps_dump/'
    pot_path = main_path   # the path for the potential
    dump_path = main_path + 'output/' + str(gb_name) + '/'
    pkl_file = main_path + 'data/' + str(gb_name) + '.pkl'
    initial_dump = main_path + 'output/' + str(gb_name) + '/dump_1'  # the name of the dump file that
    # initial_dump = main_path + 'output/' + str(gb_name) + '/dump.29'
    output = dump_path + 'dump_min'

    # For running on HPC############################################################
    # lammps_exe_path = '/usr/local/usrapps/spatala/lammps-12Dec18/src/lmp_mpi'
    # main_path = '/gpfs_common/share02/spatala/Leila_lammps/GBMC/'
    # pot_path = main_path   # the path for the potential
    # dump_path = main_path + 'output/' + str(gb_name) + '/'
    # pkl_file = main_path + 'data/' + str(gb_name) + '.pkl'
    # initial_dump = main_path + 'output/' + str(gb_name) + '/dump_1'  # the name of the dump file that
    # output =  dump_path + 'dump_min'
    #################################################################################
    #  --------------------------------------------------------------------------
    #  Create lammps dump file for pkl file
    #  --------------------------------------------------------------------------
    box_bound, dump_lamp, box_type = ldw.lammps_box(pkl_file)  # lammps creates from the pkl file
    ldw.write_lammps_dump(initial_dump, box_bound, dump_lamp, box_type)  # writing the dump file

    #  --------------------------------------------------------------------------
    # Define the path to dump files
    #  --------------------------------------------------------------------------
    filename_0 = dump_path + 'dump.0'  # the output of previous step
    # the initila minimization lammps script write the in.min script and run it and create dump_minimized
    fil_name = 'in.min_1'

    lsw.run_lammps_anneal(initial_dump, fil_name, pot_path, l1, tol_fix_reg, lammps_exe_path, filename_0, Tm,
                        "aneal", step=1, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val, MaxEval=MaxEval_val,
                        Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val, Iter_cool=Iter_cool_val)
    #  --------------------------------------------------------------------------
    #  Start Monte Carlo iterations
    #  --------------------------------------------------------------------------
    iter = 5000  # numbber of iterations
    ff = open('output', 'w')
    for i in range(0, iter, 1):
        print(i)
        #  read the data
        data_0 = uf.compute_ovito_data(filename_0)
        non_p = uf.identify_pbc(data_0)
        #  find the gb atoms
        GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data_0, l1, non_p, str_alg, csc_tol)

        #  decide between remove and insertion
        choice = uf.choos_rem_ins()
        #  --------------------------------------------------------------------------
        #  If the choice is removal
        #  --------------------------------------------------------------------------
        if choice == "removal":
            p_rm = uf.RemProb(data_0, CohEng, GbIndex)
            ID2change = uf.RemIns_decision(p_rm)
            ff.write(filename_0 + '\n')

            ff.write(str(i) + ' ' + choice + ' ' + str(GbIndex[ID2change]))
            ff.write('\n')
            print(GbIndex[ID2change])

            var2change = data_0.particles['Particle Identifier'][GbIndex[ID2change]]
            uf.atom_removal(filename_0, dump_path, GbIndex[ID2change], var2change)
            # the initila minimization lammps script write the in.min script and run it and create dump_minimized
            fil_name = 'in.min_1'
            filename_rem = dump_path + 'rem_dump'
            # copyfile(filename_rem, dump_path + 'rem/rem_dump_' + str(i))
            filename_1 = dump_path + 'dump.' + str(i)
            #  ---------------------------------------
            lsw.run_lammps_anneal(filename_rem, fil_name, pot_path, l1, tol_fix_reg, lammps_exe_path, filename_1,
                                Tm, "aneal", step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val,
                                MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val,
                                Iter_cool=Iter_cool_val)
            lines = open(filename_1, 'r').readlines()
            lines[1] = '0\n'
            out = open(filename_1, 'w')
            out.writelines(lines)
            out.close()

            #  -----------------------------------------------------------------------------

            data_1 = uf.compute_ovito_data(filename_1)
            SC_boolean = uf.check_SC_reg(data_1, l1, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol)

            if str_alg == "ptm":
                assert data_0.particles['Structure Type'][GbIndex[ID2change]] != st_id
            else:
                assert data_0.particles['c_csym'][GbIndex[ID2change]] > .1
            if SC_boolean is False:
                sim_cell = data_1.cell[...]
                uniq_atoms = uf.add_sc(pkl_file, data_1, l1, rCut, non_p, tol_fix_reg,
                                    SC_tol, str_alg, csc_tol, box_bound)
                ldw.write_lammps_dump(filename_rem, box_bound, uniq_atoms, box_type)
                lsw.run_lammps_anneal(filename_rem, fil_name, pot_path, l1, tol_fix_reg, lammps_exe_path,
                                    filename_1, Tm, "sc",  step=1, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val,
                                    MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val,
                                    Iter_cool=Iter_cool_val)

            E_1 = uf.cal_GB_E(data_1, weight_1, non_p, l1, CohEng, str_alg, csc_tol)  # after removal
            E_0 = uf.cal_GB_E(data_0, weight_1, non_p, l1, CohEng, str_alg, csc_tol)
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
            ff.write(filename_0 + '\n')
            pts_w_imgs, gb1_inds, inds_arr = pdf.pad_dump_file(data_0, l1, rCut, non_p, str_alg, csc_tol)
            tri_vertices, gb_tri_inds = vvp.triang_inds(pts_w_imgs, gb1_inds, inds_arr)
            cc_coors, cc_rad = vvp.vv_props(pts_w_imgs, tri_vertices, gb_tri_inds, l1)
            cc_coors1 = vvp.wrap_cc(data_0.cell, cc_coors)
            Prob = uf.radi_normaliz(cc_rad)
            ID2change = uf.RemIns_decision(Prob)
            pos_add_atom = cc_coors[ID2change]
            atom_id = np.max(data_0.particles['Particle Identifier']+1)
            uf.atom_insertion(filename_0, dump_path, pos_add_atom, atom_id)
            ff.write(str(i) + ' ' + choice + ' ' + str(pos_add_atom))
            ff.write('\n')

            filename_1 = dump_path + 'dump.' + str(i)
            #  -------------------------------------------------------------------------------------------------
            # the initila minimization lammps script write the in.min script and run it and create dump_minimized
            fil_name = 'in.min_1'
            filename_ins = dump_path + 'ins_dump'
            filename_1 = dump_path + 'dump.' + str(i)
            #  --------------------------------------------------------------------------------------------------

            # copyfile(filename_1, dump_path + 'ins/min1_' + str(i))
            lsw.run_lammps_anneal(filename_ins, fil_name, pot_path, l1, tol_fix_reg, lammps_exe_path, filename_1,
                                Tm, "aneal", step=2, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val,
                                MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val,
                                Iter_cool=Iter_cool_val)

            lines = open(filename_1, 'r').readlines()
            lines[1] = '0\n'
            out = open(filename_1, 'w')
            out.writelines(lines)
            out.close()
            # copyfile(out_heat, dump_path + 'ins/heat_' + str(i))
            data_1 = uf.compute_ovito_data(filename_1)
            SC_boolean = uf.check_SC_reg(data_1, l1, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol)
            if SC_boolean is False:
                sim_cell = data_1.cell[...]
                # box_bound, box_type = ldw.box_bound_func(sim_cell)
                uniq_atoms = uf.add_sc(pkl_file, data_1, l1, rCut, non_p, tol_fix_reg, SC_tol,
                                    str_alg, csc_tol, box_bound)
                ldw.write_lammps_dump(filename_ins, box_bound, uniq_atoms, box_type)
                lsw.run_lammps_anneal(filename_ins, fil_name, pot_path, l1, tol_fix_reg, lammps_exe_path,
                                    filename_1, Tm, "sc",  step=1, Etol=Etol_val, Ftol=Ftol_val, MaxIter=MaxIter_val,
                                    MaxEval=MaxEval_val, Iter_heat=Iter_heat_val, Iter_equil=Iter_equil_val,
                                    Iter_cool=Iter_cool_val)

            E_1 = uf.cal_GB_E(data_1, weight_1, non_p, l1, CohEng, str_alg, csc_tol)
            E_0 = uf.cal_GB_E(data_0, weight_1, non_p, l1, CohEng, str_alg, csc_tol)
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


element = 'Mg_sun'
CohEng = -1.5287  # calculated from in.cohesive
Tm = 923.1
lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
gb_name = 'gb_attr_Mg_sun_S7_1_N1_0_0_1_N2_0_0_-1'
MC_MD(gb_name, element, CohEng, Tm, lammps_exe_path, str_alg="ptm")