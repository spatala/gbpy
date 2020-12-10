
import numpy as np
import util_funcs as uf
import os


def file_gen(fil_name):
    """
    Function opens the input file.

    Parameters
    ------------
    fil_name : str
        name of the input lammps dump file
    Returns
    ----------
    fiw:
        the opened lammps dump file
    fil_name: str
        name of the input lammps dump file
    """
    fiw = open(fil_name, 'w')
    return fiw, fil_name


def run_lammps_anneal(filename0, fil_name, pot_path, lat_par, tol_fix_reg, lammps_exe_path,
                      output, Tm, check,   step=2, Etol=1e-25, Ftol=1e-25, MaxIter=5000, MaxEval=10000,
                      Iter_heat=1000, Iter_equil=10000, Iter_cool=12000):
    """
    Function runs the lammps script

    Parameters
    ------------
    filename0 : str
        the opened lammps dump file
    fil_name: str
        name of the input lammps dump file
    pot_path: str
        The directory of the interatomic potential
    lat_par: int
        The lattice parameter
    tol_fix_reg: int
        The thickness of the fix rigid part in the simulation cell
    lammps_exe_path: str
        The path of the lammps .exe file
    output: str
        The name of the lammps output file
    Tm: int
        The melting point of the material of interest
    Etol: int
        stopping tolerance for energy (unitless)
    Ftol: int
        stopping tolerance for force (force units)
    MaxIter: int
        max iterations of minimizer
    MaxEval: int
        max number of force/energy evaluations
    Iter_heat: int
        number of iteration for the heating process
    Iter_equil: int
        number of iteration for the equilibration process
    Iter_cool: int
        number of iteration for the cooling process

    Returns
    ----------
    """
    data = uf.compute_ovito_data(filename0)
    non_p = uf.identify_pbc(data)
    run_lmp(non_p, fil_name, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool,
            lat_par, filename0, pot_path, output, check)
    os.system('mpirun -np 2 ' + str(lammps_exe_path) + ' -in ' + fil_name)


def run_lmp(non_p, fil_name, Etol, Ftol, MaxIter, MaxEval, Iter_heat, Iter_equil, Iter_cool, lat_par,
            dump_name, pot_path, output, check):
    """
    Function writes the lammps script for heating/equil/cooling process

    Parameters
    ------------
    non_p: int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    fil_name: str
        name of the input lammps dump file
    Etol: int
        stopping tolerance for energy (unitless)
    Ftol: int
        stopping tolerance for force (force units)
    MaxIter: int
        max iterations of minimizer
    MaxEval: int
        max number of force/energy evaluations
    Iter_heat: int
        number of iteration for the heating procedure
    Iter_equil: int
        number of iteration for the equilibration procedure
    Iter_cool: int
        number of iteration for the cooling procedure
    lat_par :
        Lattice parameter for the crystal being simulated
    dump_name: str
        Name of the lammps dump file
    pot_path: str
        The directory of the interatomic potential
    output: str
        Name of the lammps script
    check: str
        If check="SC" it means the gb gets very close to the GB.

    Returns
    ----------
    """
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
    if check == "sc":
        line.append('delete_atoms overlap ${OverLap}  all all\n')
    else:
        line.append('delete_atoms overlap ${OverLap}  upper lower\n')
    line.append('# ---------- Run Minimization ---------------------\n')
    line.append('reset_timestep 0\n')

    line.append('thermo 10\n')
    line.append('thermo_modify lost ignore\n')
    line.append('min_style cg\n')
    line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval} \n')

    line.append('# ---------- Run heat --------------------- \n')
    # line.append('reset_ids\n')
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
    line.append('dump 2 all custom 1000 ' + str(output) + ' id type x y z c_csym c_eng\n')
    line.append('dump_modify 2 every 1000 sort id first yes\n')
    line.append('run 0 \n')
    line.append('undump 2\n')

    for i in line:
        fiw.write(i)

    return True
