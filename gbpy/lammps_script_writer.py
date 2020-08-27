
import numpy as np
import util_funcs as uf
import os


def file_gen(fil_name):
    fiw = open(fil_name, 'w')
    return fiw, fil_name


def lammps_script_var(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval):
    """
    Function writes the variables in the lammps script.

    Parameters
    ------------
    fiw :
        The name of the lammps script file
    lat_par :
        Lattice parameter for the crystal being simulated
    Etol :

    Ftol :

    MaxIter :

    MaxEval :


    Returns
    ----------
    """
    overlap_cte = np.sqrt(2) * lat_par / 4
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal ' + str(Etol) + '\n')
    line.append('variable Ftol equal ' + str(Ftol) + '\n')
    line.append('variable MaxIter equal ' + str(MaxIter) + '\n')
    line.append('variable MaxEval equal ' + str(MaxEval) + '\n')
    line.append('\n')
    line.append('# Structural variables------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal ' + str(lat_par) + '\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable OverLap equal ' + str(overlap_cte) + '\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def lammps_script_var_anneal(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool):
    """
    Function 

    Parameters
    ------------
    fiw :
        The name of the lammps script file
    lat_par :
        Lattice parameter for the crystal being simulated
    Etol :

    Ftol :

    MaxIter :

    MaxEval :
    Iter_heat :
    Iter_equil :
    Iter_cool :

    Returns
    ----------
    """
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal ' + str(Etol) + '\n')
    line.append('variable Ftol equal ' + str(Ftol) + '\n')
    line.append('variable MaxIter equal ' + str(MaxIter) + '\n')
    line.append('variable MaxEval equal ' + str(MaxEval) + '\n')
    line.append('variable Iter_heat equal ' + str(Iter_heat) + '\n')
    line.append('variable Iter_equil equal ' + str(Iter_equil) + '\n')
    line.append('variable Iter_cool equal ' + str(Iter_cool) + '\n')
    line.append('\n')
    line.append('# Structural variables------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal ' + str(lat_par) + '\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_init_sim(fiw, non_p):
    """
    Function 

    Parameters
    ------------
    fiw :
        The name of the lammps script file
    non_p :

    Returns
    ----------
    """
    if non_p == 0:
        bound = 'f p p'
    elif non_p == 1:
        bound = 'p f p'
    else:
        bound = 'p p f'

    line = []
    line.append('# -----------Initializing the Simulation-----------\n')
    line.append('\n')
    line.append('clear\n')
    line.append('units metal\n')
    line.append('dimension 3\n')
    line.append('boundary ' + str(bound) + '\n')
    line.append('atom_style atomic\n')
    line.append('atom_modify map array\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def define_box(fiw, untilted, tilt, box_type):
    """
    Function 

    Parameters
    ------------
    fiw :

    untilted :

    tilt :

    box_type :

    Returns
    ----------
    """
    if box_type == 'block':
        whole_box = 'region whole block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' units box'
    elif box_type == 'prism':
        whole_box = 'region whole prism ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' ' + str(tilt[0]) + ' ' + str(tilt[1]) + ' ' + str(tilt[2])\
                     + ' units box'
    create_box = 'create_box 2 whole\n'
    line = []
    line.append('# ---------Creating the Atomistic Structure--------\n')
    line.append('\n')
    line.append('lattice fcc ${LatParam}\n')
    line.append(str(whole_box) + '\n')
    line.append(str(create_box) + '\n')

    for i in line:
        fiw.write(i)

    return True


def script_pot(fiw, pot_path):
    """
    Function 

    Parameters
    ------------
    fiw :

    pot_path :

    Returns
    ----------
    """
    line = []
    line.append('# -------Defining the potential functions----------\n')
    line.append('\n')
    line.append('pair_style eam/alloy\n')
    line.append('pair_coeff * * ' + str(pot_path) + 'Al99.eam.alloy Al Al\n')

    line.append('neighbor 2 bin\n')
    line.append('neigh_modify delay 5 check yes\n')

    for i in line:
        fiw.write(i)

    return True


def script_read_dump(fiw, dump_name):
    """
    Function 

    Parameters
    ------------
    fiw :

    dump_name :

    Returns
    ----------
    """
    line = []
    line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')
    for i in line:
        fiw.write(i)

    return True


def script_overlap(fiw, untilted, tol_fix_reg, non_p, step):
    """
    Function 

    Parameters
    ------------
    fiw :

    untilted :
    tol_fix_reg :
    non_p :
    step :

    Returns
    ----------
    """
    untilted[non_p, :] = untilted[non_p, :] + np.array([-tol_fix_reg, tol_fix_reg])
    # if non_p == 0:
    #     var = 'x'
    # elif non_p == 1:
    #     var = 'y'
    # else:
    #     var = 'z'
    line = []
    line.append('group lower type 2 \n')
    line.append('group upper type 1\n')
    if step == 1:
        line.append('delete_atoms overlap ${OverLap}  upper lower\n')
        # line.append('change_box all ' + str(var) + ' final ' + str(untilted[non_p, 0])
        # + ' ' + str(untilted[non_p, 1]) + ' units box\n')

    for i in line:
        fiw.write(i)

    return True


def script_compute(fiw):
    """
    Function 

    Parameters
    ------------
    fiw :

    Returns
    ----------
    """
    line = []
    line.append('\n')
    line.append('# ---------Computing Simulation Parameters---------\n')
    line.append('\n')
    line.append('compute csym all centro/atom fcc\n')
    line.append('compute eng all pe/atom\n')
    line.append('compute eatoms all reduce sum c_eng\n')
    line.append('compute MinAtomEnergy all reduce min c_eng\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_min_sec(fiw, output, non_p, box_type):
    """
    Function 

    Parameters
    ------------
    fiw :

    output :
    non_p :
    box_type :

    Returns
    ----------
    """
    line = []
    line.append('\n')
    line.append('#----------------------Run minimization 1--------------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('min_style cg\n')
    line.append('minimize 1e-5 1e-5 5000 10000\n')

    line.append('\n')
    line.append('#----------------------Run minimization 2--------------------\n')
    line.append('\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    # if non_p == 0:
    #     if box_type == "block":
    #         line.append('fix 1 all box/relax y 0 z 0 vmax .001\n')
    #     else:
    #         line.append('fix 1 all box/relax y 0 z 0 yz 0 vmax .001\n')
    # elif non_p == 1:
    #     if box_type == "block":
    #         line.append('fix 1 all box/relax x 0 z 0 vmax .001\n')
    #     else:
    #         line.append('fix 1 all box/relax x 0 z 0 xz 0 vmax .001\n')
    # else:
    #     if box_type == "block":
    #         line.append('fix 1 all box/relax x 0 y 0 vmax .001\n')
    #     else:
    #         line.append('fix 1 all box/relax x 0 y 0 xy 0 vmax .001\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
    line.append('\n')
    line.append('#----------------------Run 0 to dump--------------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('reset_ids\n')
    line.append('thermo 100\n')
    line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
                'c_eatoms c_MinAtomEnergy\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom ${MaxIter} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${MaxIter} sort id first yes\n')
    line.append('run 0\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')

    for i in line:
        fiw.write(i)

    return True


def script_heating(fiw, output, Tm, non_p):
    """
    Function 

    Parameters
    ------------
    fiw :

    output :
    Tm :
    non_p :

    Returns
    ----------
    """
    T = Tm / 2
    line = []

    line.append('velocity all create ' + str(T) + ' 235911\n')
    line.append('fix 1 all nve \n')
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('run 10000\n')
    line.append('unfix 1\n')

    line.append('# -----------------heating step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp .01 ' + str(T) +
                    ' $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp .01 ' + str(T) +
                    ' $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp .01 ' + str(T) +
                    ' $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')

    line.append('thermo 100\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 100 ./lammps_dump/heat_0 id type x y z c_csym c_eng\n')
    line.append('run ${Iter_heat}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_equil(fiw, output, Tm, non_p):
    """
    Function 

    Parameters
    ------------
    fiw :

    output :
    Tm :
    non_p :

    Returns
    ----------
    """
    T = Tm / 2
    line = []
    line.append('# -----------------equilibrium step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.0001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) +
                    ' $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) +
                    ' $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp ' + str(T) + ' ' + str(T) +
                    ' $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')

    line.append('thermo 100\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('dump 1 all custom 10 ./lammps_dump/equil id type x y z c_csym c_eng\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_cooling(fiw, output, Tm, non_p):
    """
    Function 

    Parameters
    ------------
    fiw :

    output :
    Tm :
    non_p :

    Returns
    ----------
    """
    T = Tm / 2
    line = []
    line.append('# -----------------cooling step-----------------\n')
    line.append('\n')
    line.append('reset_timestep 0\n')
    line.append('timestep 0.001\n')
    if non_p == 0:
        line.append('fix 1 all npt temp ' + str(T) +
                    ' .01 $(100.0*dt) couple yz  y 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    elif non_p == 1:
        line.append('fix 1 all npt temp ' + str(T) +
                    ' .01 $(100.0*dt) couple xz  x 0.0 0.0 $(1000.0*dt)  z 0.0 0.0 $(1000.0*dt)\n')
    else:
        line.append('fix 1 all npt temp ' + str(T) +
                    ' .01 $(100.0*dt) couple xy  x 0.0 0.0 $(1000.0*dt)  y 0.0 0.0 $(1000.0*dt)\n')
    line.append('thermo 10\n')
    line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_cool}\n')
    line.append('dump 1 all custom ${Iter_cool} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_cool} sort id first yes\n')
    line.append('run 0\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')
    line.append('\n')
    for i in line:
        fiw.write(i)

    return True


def script_nve_nvt(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool):
    """
    Function 

    Parameters
    ------------
    fiw :

    output :
    Tm :
    non_p :
    Iter_heat :
    Iter_equil :
    Iter_cool :

    Returns
    ----------
    """

    T = Tm / 2
    line = []
    line.append('velocity all create ' + str(T) + ' 235911\n')
    line.append('fix 1 all nve \n')
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run 1000\n')
    line.append('unfix 1\n')

    line.append('# -----------------equilibrium step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp ' + str(T) + ' ' + str(T) + ' $(100.0*dt)\n')
    line.append('dump 1 all custom 100 equi id type x y z c_csym c_eng\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')
    line.append('undump 1\n')

    line.append('# -----------------cooling step-----------------\n')
    line.append('\n')

    line.append('fix 1 all nvt temp  ' + str(T) + ' .01 .01\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_cool}\n')
    line.append('dump 1 all custom ${Iter_cool} ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 1 every ${Iter_cool} sort id first yes\n')
    line.append('run 0\n')
    line.append('undump 1\n')
    line.append('\n')

    for i in line:
        fiw.write(i)

    return True


def script_main_min(fil_name, lat_par, tol_fix_reg, dump_name, pot_path, non_p, output,
                    step, Etol, Ftol, MaxIter, MaxEval):
    """
    Function 

    Parameters
    ------------
    fil_name :

    lat_par :
    tol_fix_reg :
    dump_name :
    pot_path :
    non_p :
    output :
    step :
    Etol :
    Ftol :
    MaxIter :
    MaxEval :

    Returns
    ----------
    """
    fiw, file_name = file_gen(fil_name)
    lammps_script_var(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval)
    script_init_sim(fiw, non_p)
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    define_box(fiw, untilted, tilt, box_type)
    script_read_dump(fiw, dump_name)
    script_pot(fiw, pot_path)
    script_overlap(fiw, untilted, tol_fix_reg, non_p, step)
    script_compute(fiw)
    script_min_sec(fiw, output, non_p, box_type)


def run_lammps_min(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path,
                   output, step=2, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000):
    """
    Function 

    Parameters
    ------------
    filename0 :
    fil_name :
    pot_path :
    lat_par :
    tol_fix_reg :
    lammps_exe_path :
    output :
    step :
    Etol :
    Ftol :
    MaxIter :
    MaxEval :

    Returns
    ----------
    """
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    script_main_min(fil_name, lat_par, tol_fix_reg, filename0, pot_path, non_p,
                    output, step, Etol, Ftol, MaxIter, MaxEval)
    os.system('mpirun -np 2 ' + str(lammps_exe_path) + ' -in ' + fil_name)


def script_main_anneal(fil_name, lat_par, tol_fix_reg, dump_name, pot_path, non_p, output,
                       Tm, step, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool):
    """
    Function 

    Parameters
    ------------
    fil_name :
    lat_par :
    tol_fix_reg :
    dump_name :
    pot_path :
    non_p :
    output :
    Tm :
    step :
    Etol :
    Ftol :
    MaxIter :
    MaxEval :
    Iter_heat :
    Iter_equil :
    Iter_cool :

    Returns
    ----------
    """
    fiw, file_name = file_gen(fil_name)
    lammps_script_var_anneal(fiw, lat_par, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool)
    script_init_sim(fiw, non_p)
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    define_box(fiw, untilted, tilt, box_type)
    script_read_dump(fiw, dump_name)
    script_pot(fiw, pot_path)
    script_compute(fiw)
    script_nve_nvt(fiw, output, Tm, non_p, Iter_heat, Iter_equil, Iter_cool)


def run_lammps_anneal(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path,
                      output, Tm,  step=2, Etol=1e-25, Ftol=1e-25, MaxIter=5000, MaxEval=10000,
                      Iter_heat=1000, Iter_equil=10000, Iter_cool=12000):
    """
    Function 

    Parameters
    ------------
    filename0 :
    fil_name :
    pot_path :
    lat_par :
    tol_fix_reg :
    lammps_exe_path :
    output :
    Tm :
    step :
    Etol :
    Ftol :
    MaxIter :
    MaxEval :
    Iter_heat :
    Iter_equil :
    Iter_cool :

    Returns
    ----------
    """
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    run_lmp(non_p, fil_name, Etol, Ftol, MaxIter, MaxEval,Iter_heat, Iter_equil, Iter_cool,lat_par, filename0, pot_path, output )
    os.system('mpirun -np 2 ' + str(lammps_exe_path) + ' -in ' + fil_name)

def run_lmp(non_p, fil_name, Etol, Ftol, MaxIter, MaxEval,Iter_heat, Iter_equil, Iter_cool,lat_par, dump_name, pot_path,output):
    fiw, file_name = file_gen(fil_name)
    if non_p == 0:
        bound = 'f p p'
    elif non_p == 1:
        bound = 'p f p'
    else:
        bound = 'p p f'
    box_bound = uf.box_size_reader(dump_name)
    untilted, tilt, box_type = uf.define_bounds(box_bound)
    overlap_cte = np.sqrt(2) * lat_par / 4
    line = []
    line.append('# Minimization Parameters -------------------------\n')
    line.append('\n')
    line.append('variable Etol equal ' + str(Etol) + '\n')
    line.append('variable Ftol equal ' + str(Ftol) + '\n')
    line.append('variable MaxIter equal ' + str(MaxIter) + '\n')
    line.append('variable MaxEval equal ' + str(MaxEval) + '\n')
    line.append('variable Iter_heat equal ' + str(Iter_heat) + '\n')
    line.append('variable Iter_equil equal ' + str(Iter_equil) + '\n')
    line.append('variable Iter_cool equal ' + str(Iter_cool) + '\n')
    line.append('\n')
    line.append('# Structural variables------------------------------\n')
    line.append('\n')
    line.append('variable LatParam equal ' + str(lat_par) + '\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable cnt equal 1\n')
    line.append('# ------------------------------------------------\n')
    line.append('\n')
    line.append('variable OverLap equal ' + str(overlap_cte) + '\n')
    line.append('\n')



    line.append('# -----------Initializing the Simulation-----------\n')
    line.append('\n')
    line.append('clear\n')
    line.append('units metal\n')
    line.append('dimension 3\n')
    line.append('boundary ' + str(bound) + '\n')
    line.append('atom_style atomic\n')
    line.append('atom_modify map array\n')
    line.append('box tilt large\n')
    if box_type == 'block':
        whole_box = 'region whole block ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' units box'
    elif box_type == 'prism':
        whole_box = 'region whole prism ' + str(untilted[0][0]) + ' ' + str(untilted[0][1]) + ' ' +\
                     str(untilted[1][0]) + ' ' + str(untilted[1][1]) + ' ' + str(untilted[2][0]) + ' '\
                     + str(untilted[2][1]) + ' ' + str(tilt[0]) + ' ' + str(tilt[1]) + ' ' + str(tilt[2])\
                     + ' units box'

    create_box = 'create_box 2 whole\n'


    line.append('compute csym all centro/atom fcc\n')
    line.append('compute eng all pe/atom\n')
    line.append('compute eatoms all reduce sum c_eng\n')
    line.append('compute MinAtomEnergy all reduce min c_eng\n')
    line.append('# ---------Creating the Atomistic Structure--------\n')
    line.append('\n')
    line.append('lattice fcc ${LatParam}\n')
    line.append(str(whole_box) + '\n')
    line.append(str(create_box) + '\n')
    line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')

    line.append('# -------Defining the potential functions----------\n')
    line.append('\n')
    line.append('pair_style eam/alloy\n')
    line.append('pair_coeff * * ' + str(pot_path) + 'Al99.eam.alloy Al Al\n')

    line.append('neighbor 2 bin\n')
    line.append('neigh_modify delay 10 check yes\n')
    line.append('group lower type 2 \n')
    line.append('group upper type 1\n')
    line.append('delete_atoms overlap ${OverLap}  upper lower\n')
    line.append('# ---------- Run Minimization ---------------------\n') 
    line.append('reset_timestep 0\n')

    line.append('thermo 10\n')
    line.append('thermo_modify lost ignore\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval} \n')

    line.append('# ---------- Run heat --------------------- \n')
    line.append('reset_ids\n')
    line.append('velocity all create 466.75 235911\n')
    line.append('fix 1 all nve \n')
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run 1000\n')
    line.append('unfix 1\n')

    line.append('# ---------- Run anneal ---------------------\n')
    line.append('fix 1 all nvt temp 466.75 466.75 $(100.0*dt)\n')
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_equil}\n')
    line.append('unfix 1\n')

    line.append('# ---------- Run cool ---------------------\n')
    line.append('fix 1 all nvt temp  466.75 .01 .01\n')
    line.append('thermo 100\n')
    line.append('thermo_modify flush yes\n')
    line.append('thermo_modify lost ignore\n')
    line.append('run ${Iter_cool}\n')
    line.append('unfix 1\n')

    line.append('#----------------------Run minimization 1--------------------\n')
    line.append('reset_timestep 0\n')
    line.append('thermo 100\n')
    line.append('thermo_modify lost ignore\n')
    line.append('min_style cg\n')
    line.append('minimize 1e-25 1e-25 5000 10000\n')

    line.append('#----------------------Run minimization 2--------------------\n') 
    line.append('reset_timestep 0 \n')
    line.append('reset_ids\n')
    line.append('thermo 100\n')  
    line.append('thermo_modify lost ignore\n')
    if non_p == 0:
        if box_type == "block":
            line.append('fix 1 all box/relax y 0 z 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax y 0 z 0 yz 0 vmax .001\n')
    elif non_p == 1:
        if box_type == "block":
            line.append('fix 1 all box/relax x 0 z 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax x 0 z 0 xz 0 vmax .001\n')
    else:
        if box_type == "block":
            line.append('fix 1 all box/relax x 0 y 0 vmax .001\n')
        else:
            line.append('fix 1 all box/relax x 0 y 0 xy 0 vmax .001\n')
    line.append('dump 2 all custom 1000 ' + str(output) +  ' id type x y z c_csym c_eng\n')   
    line.append('dump_modify 2 every 1000 sort id first yes\n')
    line.append('run 0 \n')   
    line.append('undump 2\n')
     
    #fix 1 all box/relax y 0.0 vmax 0.001
    
    for i in line:
        fiw.write(i)

    return True

# lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
# lat_par = 4.05
# tol_fix_reg = lat_par * 5
# filename0 = './dump.1'
# fil_name = 'in.min_1'
# pot_path = './lammps_dump/'
# Tm = 1000
# out_heat = './dump.heat'
# run_lammps_anneal(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
#                    Tm,  step=2, Etol=1e-25, Ftol=1e-25, MaxIter=5000, MaxEval=10000, Iter_heat=1000,
#                    Iter_equil=3000, Iter_cool=1000)

# fil_name1 = 'in.anneal'
# out_heat = './lammps_dump/heat'
# run_lammps_anneal(out_min_1, fil_name1, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_heat,\
#                    Tm,  step=2, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000, Iter_heat=1000,
#                    Iter_equil=20000, Iter_cool=10000)

# lines = open(out_heat, 'r').readlines()
# lines[1] = '0\n'
# out = open(out_heat, 'w')
# out.writelines(lines)
# out.close()


# fil_name2 = 'in.min_2'
# out_min_2 = './lammps_dump/final'
# run_lammps_min(out_heat, fil_name2, pot_path, lat_par, tol_fix_reg, lammps_exe_path, out_min_2,\
#                step=2, Etol=1e-25, Ftol=1e-25, MaxIter=10000, MaxEval=10000)
