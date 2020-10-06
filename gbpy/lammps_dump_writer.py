import numpy as np
import pickle as pkl


def lammps_box(lat_par, pkl_name):
    """
    Function calculates the box bound and the atom coordinates of the GB simulation.
    Ref: https://lammps.sandia.gov/doc/Howto_triclinic.html

    Parameters
    ------------
    pkl_name :
        The name of the pkl file which contains the simulation cell ( a 3*4 numpy array
        where the first 3 columns are the cell vectors and the last column is the box origin),
        the cordinates of the upper and lower grain.

    Returns
    ----------
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz
    dump_lamp :
        A numpy nd.array having atom ID, atom type( 1 for upper grain and 2 for lower grain), x, y, z
    box_type :
        The type of simulation box which is eaither "prism" or "block"
    """

    jar = open(pkl_name, 'rb')
    gb_attr = pkl.load(jar)
    jar.close()

    u_pts = gb_attr['upts']
    len_u = np.shape(u_pts)[0]
    u_type = np.zeros((len_u, 1)) + 1

    l_pts = gb_attr['lpts']
    len_l = np.shape(l_pts)[0]
    l_type = np.zeros((len_l, 1)) + 2

    sim_cell = gb_attr['cell']
    # origin_o = sim_cell[:, 3]

    # # “origin” at (xlo,ylo,zlo)
    # xlo = origin_o[0]
    # ylo = origin_o[1]
    # zlo = origin_o[2]
    # # a = (xhi-xlo,0,0);
    # xhi = sim_cell[0, 0] + xlo
    # # b = (xy,yhi-ylo,0);
    # xy = sim_cell[0, 1]
    # yhi = sim_cell[1, 1] + ylo
    # # c = (xz,yz,zhi-zlo)
    # xz = sim_cell[0, 2]
    # yz = sim_cell[1, 2]
    # zhi = sim_cell[2, 2] + zlo

    # if xy or xz or yz != 0:
    #     box_type = "prism"
    # else:
    #     box_type = "block"

    # xlo_bound = xlo + np.min(np.array([0, xy, xz, xy + xz]))
    # xhi_bound = xhi + np.max(np.array([0, xy, xz, xy + xz]))
    # ylo_bound = ylo + np.min(np.array([0, yz]))
    # yhi_bound = yhi + np.max(np.array([0, yz]))
    # zlo_bound = zlo
    # zhi_bound = zhi

    upper = np.concatenate((u_type, u_pts), axis=1)
    lower = np.concatenate((l_type, l_pts), axis=1)

    all_atoms = np.concatenate((lower, upper))
    num_atoms = len_u + len_l
    ID = np.arange(num_atoms).reshape(num_atoms, 1) + 1
    dump_lamp = np.concatenate((ID, all_atoms), axis=1)
    box_bound, box_type = box_bound_func(sim_cell)

    # if box_type == "block":
    #     box_bound = np.array([[xlo_bound, xhi_bound], [ylo_bound, yhi_bound], [zlo_bound, zhi_bound]])
    # else:
    #     box_bound = np.array([[xlo_bound, xhi_bound, xy], [ylo_bound, yhi_bound,  xz], [zlo_bound, zhi_bound, yz]])

    return box_bound, dump_lamp, box_type

def box_bound_func(sim_cell):
    origin_o = sim_cell[:, 3]
    xlo = origin_o[0]
    ylo = origin_o[1]
    zlo = origin_o[2]

    xhi = sim_cell[0, 0] + xlo
    xy = sim_cell[0, 1]
    yhi = sim_cell[1, 1] + ylo
    xz = sim_cell[0, 2]
    yz = sim_cell[1, 2]
    zhi = sim_cell[2, 2] + zlo

    if xy or xz or yz != 0:
        box_type = "prism"
    else:
        box_type = "block"

    xlo_bound = xlo + np.min(np.array([0, xy, xz, xy + xz]))
    xhi_bound = xhi + np.max(np.array([0, xy, xz, xy + xz]))
    ylo_bound = ylo + np.min(np.array([0, yz]))
    yhi_bound = yhi + np.max(np.array([0, yz]))
    zlo_bound = zlo
    zhi_bound = zhi

    if box_type == "block":
        box_bound = np.array([[xlo_bound, xhi_bound], [ylo_bound, yhi_bound], [zlo_bound, zhi_bound]])
    else:
        box_bound = np.array([[xlo_bound, xhi_bound, xy], [ylo_bound, yhi_bound,  xz], [zlo_bound, zhi_bound, yz]])
    return box_bound, box_type


def write_lammps_dump(filename0, box_bound, dump_lamp, box_type):
    """
    Function writes the lammps dump file.

    Parameters
    ------------
    filename0 :
        Name of the lammps dump file
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo, yhi,
        zlo, zhi, xy, xz, yz
    dump_lamp :
        A numpy nd.array having atom ID, atom type( 1 for upper grain and 2 for lower grain), x, y, z

    Returns
    ----------
    """
    p_x = box_bound[0,1] - box_bound[0,0]
    p_y = box_bound[1,1] - box_bound[1,0]
    p_z = box_bound[2,1] - box_bound[2,0]

    non_p_dir = np.argmax([p_x, p_y, p_z])

    num_atoms = np.shape(dump_lamp)[0]
    file = open(filename0, "w")
    file.write("ITEM: TIMESTEP\n")
    file.write("0\n")
    file.write("ITEM: NUMBER OF ATOMS\n")
    file.write(str(num_atoms) + "\n")
    # file.write("ITEM: BOX BOUNDS xy xz yz pp ff pp\n")
    if box_type == "prism":
        if non_p_dir == 0:
            file.write("ITEM: BOX BOUNDS xy xz yz ff pp pp\n")
        elif non_p_dir ==1:
            file.write("ITEM: BOX BOUNDS xy xz yz pp ff pp\n")
        else:
            file.write("ITEM: BOX BOUNDS xy xz yz pp pp ff\n")
        
    else:
        if non_p_dir == 0:
            file.write("ITEM: BOX BOUNDS ff pp pp\n")
        elif non_p_dir ==1:
            file.write("ITEM: BOX BOUNDS pp ff pp\n")
        else:
            file.write("ITEM: BOX BOUNDS pp pp ff\n")
        
    file.write(' '.join(map(str, box_bound[0])) + "\n")
    file.write(' '.join(map(str, box_bound[1])) + "\n")
    file.write(' '.join(map(str, box_bound[2])) + "\n")
    file.write("ITEM: ATOMS id type x y z\n")
    file.close()
    mat = np.matrix(dump_lamp)
    with open(filename0, 'a') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d %d %.10f %.10f %.10f')


# def write_lammps_script(lat_par, dump_name, path, script_name,  box_bound, box_type, tol_fix_reg):

#     """
#     Function writes the lammps script to minimize the simulation box.

#     Parameters
#     ------------
#     dump_name :
#         Name of the lammps dump file.
#     path :
#         The path that the lammps dump files will be saved.
#     script_name :
#         The name of the lammps script created for minimization.
#     box_bound :
#         The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo, yhi,
#         zlo, zhi, xy, xz, yz
#     box_type :
#         The type of simulation box which is eaither "prism" or "block"

#     Returns
#     ----------
#     """
#     # the voundaries of change_box
#     delta_low = -tol_fix_reg
#     delta_high = tol_fix_reg
#     # boundaries of fix rigid
#     xy = box_bound[0, 2]
#     xz = box_bound[1, 2]
#     yz = box_bound[2, 2]
#     xlo = box_bound[0, 0] - np.min(np.array([0, xy, xz, xy + xz]))
#     xhi = box_bound[0, 1] - np.max(np.array([0, xy, xz, xy + xz]))
#     ylo = box_bound[1, 0] - np.min(np.array([0, yz]))
#     yhi = box_bound[1, 1] - np.max(np.array([0, yz]))
#     zlo = box_bound[2, 0]
#     zhi = box_bound[2, 1]
#     z0 = zlo + tol_fix_reg
#     z1 = zhi - tol_fix_reg

#     fiw = open(str(path) + str(script_name), 'w')
#     line = []
#     line.append('# Minimization Parameters -------------------------\n')
#     line.append('\n')
#     line.append('variable Etol equal 1e-25\n')
#     line.append('variable Ftol equal 1e-25\n')
#     line.append('variable MaxIter equal 5000\n')
#     line.append('variable MaxEval equal 10000\n')
#     line.append('variable MaxEval0 equal 1000\n')
#     line.append('variable MaxEval1 equal 12000\n')
#     line.append('# ------------------------------------------------\n')
#     line.append('\n')
#     line.append('variable LatParam equal ' + str(lat_par) + '\n')
#     line.append('# ------------------------------------------------\n')
#     line.append('\n')
#     line.append('variable GBname index in.minimize0\n')
#     line.append('variable cnt equal 1\n')
#     line.append('# ------------------------------------------------\n')
#     line.append('\n')
#     line.append('variable OverLap equal 1.43\n')
#     line.append('\n')
#     line.append('# -----------Initializing the Simulation-----------\n')
#     line.append('\n')
#     line.append('clear\n')
#     line.append('units metal\n')
#     line.append('dimension 3\n')
#     line.append('boundary p p f \n')
#     line.append('atom_style atomic\n')
#     line.append('\n')
#     line.append('# ---------Creating the Atomistic Structure--------\n')
#     line.append('\n')
#     line.append('lattice fcc ${LatParam}\n')
#     if box_type == "prism":
#         line.append('region whole prism ' + str(xlo) + ' ' +
#                     str(xhi) + ' ' + str(ylo) + ' ' + str(yhi) + ' ' +
#                     str(zlo) + ' ' + str(zhi) + ' ' + str(xy)
#                     + ' ' + str(xz) + ' ' + str(yz) + ' units box\n')

#         line.append('region reg_fix prism ' + str(xlo) + ' ' +
#                     str(xhi) + ' ' + str(ylo) + ' ' + str(yhi) + ' ' +
#                     str(z0) + ' ' + str(z1) + ' ' + str(xy)
#                     + ' ' + str(xz) + ' ' + str(yz) + ' units box\n')
#     else:
#         line.append('region whole block ' + str(xlo) + ' ' +
#                     str(xhi) + ' ' + str(ylo) + ' ' + str(yhi) + ' ' +
#                     str(zlo) + ' ' + str(zhi) + ' ' + ' units box\n')

#         line.append('region reg_fix block ' + str(xlo) + ' ' +
#                     str(xhi) + ' ' + str(ylo) + ' ' + str(yhi) + ' ' +
#                     str(z0) + ' ' + str(z1) + ' units box\n')

#     line.append('create_box 2 whole\n')
#     line.append('read_dump ' + str(dump_name) + ' 0 x y z box yes add yes \n')
#     line.append('\n')

#     line.append('group lower type 2 \n')
#     line.append('group upper type 1\n')

#     line.append("#---------Defining the fix rigid region--------------\n")
#     line.append('group non_fix region reg_fix\n')
#     line.append('group fix_reg subtract all non_fix\n')

#     line.append('\n')
#     line.append('# -------Defining the potential functions----------\n')
#     line.append('\n')
#     line.append('pair_style eam/alloy\n')
#     line.append('pair_coeff * * ' + str(path) + 'Al99.eam.alloy Al Al\n')
#     line.append('delete_atoms overlap ${OverLap}  upper lower\n')

#     line.append('neighbor 2 bin\n')
#     line.append('neigh_modify delay 10 check yes\n')
#     line.append('change_box all z delta ' + str(delta_low) + ' ' + str(delta_high) + '\n')

#     line.append('\n')
#     line.append('# ---------Computing Simulation Parameters---------\n')
#     line.append('\n')
#     line.append('compute csym all centro/atom fcc\n')
#     line.append('compute eng all pe/atom\n')
#     line.append('compute eatoms all reduce sum c_eng\n')
#     line.append('compute MinAtomEnergy all reduce min c_eng\n')
#     line.append('\n')
#     line.append('# ------1st Minimization:Relaxing the bi-crystal------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('thermo 100\n')
#     line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi ' +
#                 'press pxx pyy pzz c_eatoms c_MinAtomEnergy\n')

#     line.append('thermo_modify lost ignore\n')

#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_befor.${cnt} id type x y z c_csym c_eng\n')
#     if box_type == "prism":
#         line.append('fix 1 all box/relax x 0 y 0 xy 0\n')
#     else:
#         line.append('fix 1 all box/relax x 0 y 0\n')
#     line.append('fix 2 fix_reg rigid single reinit yes\n')

#     line.append('min_style cg\n')
#     line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
#     line.append('undump 1\n')
#     line.append('unfix 1\n')
#     line.append('\n')

#     line.append('# -----------------heating step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp .1 466.75 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 10\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_1 id type x y z c_csym c_eng\n')
#     line.append('run ${MaxEval0}\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')

#     line.append('# -----------------equilibrium step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp 466.75 466.75 0.1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 10\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_2 id type x y z c_csym c_eng\n')
#     line.append('run ${MaxEval}\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')
#     line.append('# -----------------cooling step-----------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('timestep 0.001\n')
#     line.append('fix 1 all npt temp 466.75 .1 .1 couple xy  x 0.0 0.0 1.0  y 0.0 0.0 1.0\n')
#     line.append('thermo 10\n')
#     line.append('thermo_style custom step temp pe lx ly lz press pxx pyy pzz c_eatoms\n')
#     line.append('dump 1 all custom 10 ' + str(path) + 'dump_step_3 id type x y z c_csym c_eng\n')
#     line.append('run ${MaxEval1}\n')
#     line.append('unfix 1\n')
#     line.append('undump 1\n')
#     line.append('\n')
#     line.append('#----------------------minimization--------------------\n')
#     line.append('\n')
#     line.append('reset_timestep 0\n')
#     line.append('thermo 100\n')
#     line.append('thermo_style custom step pe lx ly lz xy xz yz xlo xhi ylo yhi zlo zhi press pxx pyy pzz '
#                 'c_eatoms c_MinAtomEnergy\n')
#     line.append('thermo_modify lost ignore\n')
#     line.append('dump 1 all custom 10 ./lammps_dump/dump_minimized id type x y z c_csym c_eng\n')
#     line.append('fix 1 all box/relax x 0 y 0 xy 0\n')
#     # line.append('fix 2 fix_reg rigid single reinit yes\n')
#     line.append('min_style cg\n')
#     line.append('minimize ${Etol} ${Ftol} ${MaxIter} ${MaxEval}\n')
#     line.append('undump 1\n')
#     line.append('unfix 1\n')
#     for i in line:
#         fiw.write(i)
#     fiw.close()
#     return True

# lat_par = 4.05
# tol_fix_reg = 5 * lat_par
# box_bound, dump_lamp, box_type = lammps_box(lat_par, './tests/data/gb_attr.pkl')
# write_lammps_dump("./tests/data/dump_1", box_bound, dump_lamp, box_type)
# write_lammps_script(4.05, './tests/data/dump_1', './lammps_dump/',
#                     'in.minimize0', box_bound, box_type, tol_fix_reg)
# lammps_exe_path = '/home/leila/Downloads/mylammps/src/lmp_mpi'
# os.system(str(lammps_exe_path) + '< ./lammps_dump/' + 'in.minimize0')
