import numpy as np
import pickle as pkl


def lammps_box(pkl_name):
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

    upper = np.concatenate((u_type, u_pts), axis=1)
    lower = np.concatenate((l_type, l_pts), axis=1)

    all_atoms = np.concatenate((lower, upper))
    num_atoms = len_u + len_l
    ID = np.arange(num_atoms).reshape(num_atoms, 1) + 1
    dump_lamp = np.concatenate((ID, all_atoms), axis=1)
    box_bound, box_type = box_bound_func(sim_cell)

    return box_bound, dump_lamp, box_type


def box_bound_func(sim_cell):
    """
    This function finds the simulation cell type and the bounds of the
    simulation cell.

    Parameters
    ------------
    sim_cell: numpy.ndarray
    A 3x4 matrix (with column-major ordering). The first
    three columns of the matrix represent the three cell
    vectors and the last column is the position of the cell’s origin.

    Returns
    ----------
    box_bound :
        The box bound needed to write lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz
    box_type :
        The type of simulation box which is eaither "prism" or "block"
    """
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
    p_x = box_bound[0, 1] - box_bound[0, 0]
    p_y = box_bound[1, 1] - box_bound[1, 0]
    p_z = box_bound[2, 1] - box_bound[2, 0]

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
        elif non_p_dir == 1:
            file.write("ITEM: BOX BOUNDS xy xz yz pp ff pp\n")
        else:
            file.write("ITEM: BOX BOUNDS xy xz yz pp pp ff\n")

    else:
        if non_p_dir == 0:
            file.write("ITEM: BOX BOUNDS ff pp pp\n")
        elif non_p_dir == 1:
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


