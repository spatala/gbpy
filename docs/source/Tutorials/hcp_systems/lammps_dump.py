import numpy as np

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
    file.write("ITEM: ATOMS id x y z\n")
    file.close()
    mat = np.matrix(dump_lamp)
    with open(filename0, 'a') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d %.10f %.10f %.10f')