import ovito.io as oio
import ovito.modifiers as ovm
from itertools import islice
import pad_dump_file as pdf
import bisect
import pickle as pkl
import numpy as np


def compute_ovito_data(filename0):
    """
    Computes the attributes of ovito

    Parameters
    ------------
    filename0 : string
        The name of the input file.

    Returns
    --------
    data : class
        all the attributes of data
    """
    pipeline = oio.import_file(filename0, sort_particles=True)
    dmod = ovm.PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1)
    pipeline.modifiers.append(dmod)
    data = pipeline.compute()
    return data


def identify_pbc(data):
    """
    Function finds the non-periodic direction

    Parameters
    ------------
    data : class
        all the attributes of data

    Returns
    --------
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    """
    pbc = data.cell.pbc
    pbc = np.asarray(pbc) + 0
    non_pbc = np.where(pbc == 0)[0][0]
    return non_pbc


def box_size_reader(dump_name):
    """
    Function reads the box_bound from lammps dump file.

    Parameters
    ------------
    dump_name : string
        The name of the lammps input file.

    Returns
    --------
    box_bound : np.array
        The box bound read from lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz

    """
    with open(dump_name) as lines:
        box_bound = np.genfromtxt(islice(lines, 5, 8))
    return box_bound


def define_bounds(box_bound):
    """
    Function to find the untilted, tilt, and box type.

    Parameters
    ------------
    box_bound : np.array
        The box bound read from lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz

    Returns
    --------
    untilted : np.array
        values of xlo, xhi, ylo, yhi, zlo, zhi.
    tilt : np.array
        For box type "block" this values id [].
        For box type "prism" this value is [xy, yz, xz].
    box_type : string
        type of box which is eaither "block" or "prism".
    """
    # boundaries of fix rigid
    siz_box = np.shape(box_bound)[1]
    if siz_box == 2:
        box_type = "block"
        xy = 0
        xz = 0
        yz = 0
        tilt = []
    else:
        box_type = "prism"
        xy = box_bound[0, 2]
        xz = box_bound[1, 2]
        yz = box_bound[2, 2]
        tilt = np.array([xy, xz, yz])
    xlo = box_bound[0, 0] - np.min(np.array([0, xy, xz, xy + xz]))
    xhi = box_bound[0, 1] - np.max(np.array([0, xy, xz, xy + xz]))
    ylo = box_bound[1, 0] - np.min(np.array([0, yz]))
    yhi = box_bound[1, 1] - np.max(np.array([0, yz]))
    zlo = box_bound[2, 0]
    zhi = box_bound[2, 1]
    untilted = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])

    return untilted, tilt, box_type


def RemProb(data, CohEng, GbIndex):
    """
    The function finds The atomic removal probabilty.

    Parameters
    --------------
    filename0 : string
        The lammps dump file
    CohEng	: float
        The cohesive energy
    GbIndex : numpy.ndarray
        The index of atoms in GB

    Return
    ----------------
    AtomicRemProb : float
        The probabilty of removing an atom
    """

    GbAtomicEng = data.particle_properties['c_eng'][GbIndex]
    Excess_Eng = (GbAtomicEng - CohEng)
    Excess_Eng[Excess_Eng < 0] = 0
    Excess_Eng_Tot = np.sum(Excess_Eng)
    p_rm = Excess_Eng/Excess_Eng_Tot
    return p_rm


def RemIns_decision(p_rm):
    """
    The function finds The atomic removal probabilty.

    Parameters
    --------------
    p_rm : numpy.ndarray
        The probability of removing an atom

    Return
    ----------------
    ID2change : integer
        The ID of the atom which will be removed/inserted
    """
    CS_prob = np.cumsum(p_rm)
    rand_num = np.random.uniform(0, 1)
    location = (bisect.bisect_left(CS_prob, rand_num))
    if CS_prob[location] == rand_num or location == 0:
        ID2change = location
    else:
        ID2change = location - 1
    return ID2change


def radi_normaliz(cc_rad):
    """
    The function finds the atomic insertion probabilty.

    Parameters
    --------------
    cc_rad :
        The circum-radius of the tetrahedrons.

    Return
    ----------------
    rad_norm :
        Probability of inserting an atom in every tetrahedron.
    """
    rad_norm = cc_rad / np.sum(cc_rad)
    return rad_norm


def choos_rem_ins():
    """
    The function makes the decision whether the trail operation is insertion or removal.

    Parameters
    --------------

    Return
    ----------------
    decision : string
        removal or insertion
    """
    rand_num = np.random.uniform(0, 1)
    if rand_num > 0.5:
        return "removal"
    else:
        return "insertion"


def atom_insertion(filename0, path2dump, cc_coors1, atom_id):
    """
    The function adds the inserted atom to the lammps dump file.

    Parameters
    --------------
    filename0 : string
        The initial lammps dump file
    path2dump : string
        The path to dump the lammps dump file
    cc_coors1 : numpy.ndarray
        The coordinates of the circum-center of the tetrahedrons.
    atom_id : numpy.ndarray
        The atom ID of the inserted atom which is the ID of last atom + 1

    Return
    ----------------
    """
    lines = open(filename0, 'r').readlines()
    lines[1] = '0\n'
    lines[3] = str(int(lines[3]) + 1)
    new_line = str(atom_id) + ' 1 ' + str(cc_coors1[0]) + ' ' + str(cc_coors1[1]) + ' '
    + str(cc_coors1[2]) + ' .1 .2\n'
    lines[3] = lines[3] + '\n'

    out = open(path2dump + 'ins_dump', 'w')
    out.writelines(lines)
    out.writelines(new_line)
    out.close()


def atom_removal(filename0, path2dump, ID2change, var):
    """
    The function removes the chosen atom from the lammps dump file.

    Parameters
    --------------
    filename0 : string
        The lammps dump file
    path2dump : string
        The path to dump the lammps dump file
    ID2change : int
        The ID of the atom which will be removed
    var : numpy.ndarray
        The indices of the atom which will be removed.
        This is just to check.

    Return
    ----------------
    """
    lines = open(filename0, 'r').readlines()
    lines[1] = '0\n'  # step should be 0
    lines[3] = str(int(lines[3]) - 1) + '\n'
    assert lines[ID2change + 9].split(" ", 1)[0] == str(var)
    lines[ID2change + 9] = ''  # 8 for the number of lines on the header

    out = open(path2dump + 'rem_dump', 'w')
    out.writelines(lines)
    out.close()


def cal_area(data, non_p):
    """
    The function finds the area of the GB plane.

    Parameters
    --------------
    data : class
        all the attributes of data
    non_p : integer
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.

    Return
    ----------------
    area : float
        The surface area of the GB plane.
    """
    sim_cell = data.cell
    arr0 = pdf.p_arr(non_p)
    area = np.linalg.norm(np.cross(sim_cell[:, arr0[0]], sim_cell[:, arr0[1]]))
    return area  # in A


def cal_GB_E(data, weight_1, non_p, lat_par, CohEng, str_alg, csc_tol):
    """
    The function finds the energy of the GB.

    Parameters
    --------------
    data : class
        all the attributes of data
    non_p :
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    lat_par :
        Lattice parameter for the crystal being simulated.
    CohEng : float
        The cohesive energy
    str_alg : string
        The algorithm used to find the atoms in the GB and the surfaces.
        str_alg="csc" which uses centrosymmetry parameter
        str_alg="ptm" which uses polyhedral template matching
    csc_tol : float
        The tolerance for identifing the atoms in the GB and surfaces using
        the str_alg="csc"
    Return
    ----------------
    E_GB : float
        The enrgy of GB plane
    """
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p, str_alg, csc_tol)

    top_min = GbRegion[1]
    top_max = GbRegion[1] + w_top_SC

    max_pos_area = weight_1 * (top_max - top_min) + top_min

    bot_min = GbRegion[0] - w_bottom_SC
    bot_max = GbRegion[0]

    min_pos_area = weight_1 * (bot_max - bot_min) + bot_min

    position_np = data.particles['Position'][...][:, non_p]
    atom_id = np.where((position_np > min_pos_area) & (position_np < max_pos_area))[0]
    E_excess = data.particles['c_eng'][...][atom_id] - CohEng
    area = cal_area(data, non_p)
    E_GB = 16021.7733 * np.sum(E_excess) / area  # convert to mj/m^2
    return E_GB


def p_boltz_func(dE, area, Tm):
    """
    The function finds the botzman probability of acceptance.

    Parameters
    --------------
    dE : float
        The energy difference the initial structure and the structure after the trial operation.
    area : float
        The surface area of the GB plane.
    Tm : float
        The melting temperature of the material

    Return
    ----------------
    p_boltz : float
        The boltzman probaility
    """
    T = Tm / 2  # in K
    kb = 1.3806485279 * 10e-23 * 1e3  # mj/K
    dE = dE * area * 1e-20
    p_boltz = np.exp(-dE / kb / T)
    return p_boltz


def decide(p_boltz):
    """
    The function decides whether the new structure is accepted or not.

    Parameters
    --------------
    p_boltz : float
        The boltzman probaility

    Return
    ----------------
    decision : string
        The decision is either "accept" or "reject"
    """
    rand_num = np.random.uniform(0, 1)
    if rand_num > p_boltz:
        return "reject"
    else:
        return "accept"


def check_SC_reg(data, lat_par, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol):
    """
    Function to identify whether single crystal region on eaither side of the GB is
    bigger than a tolerance (SC_tol)

    Parameters
    ------------
    data :
        Data object computed using OVITO I/O
    lat_par :
        Lattice parameter for the crystal being simulated
    rCut :
        Cut-off radius for computing Delaunay triangulations
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    tol_fix_reg : float
        The user defined tolerance for the size of rigid translation region in lammps simulation.
    SC_tol : float
        The user defined tolerance for the minimum size of single crystal region.
    str_alg : string
        The algorithm used to find the atoms in the GB and the surfaces.
        str_alg="csc" which uses centrosymmetry parameter
        str_alg="ptm" which uses polyhedral template matching
    csc_tol : float
        The tolerance for identifing the atoms in the GB and surfaces using
        the str_alg="csc"

    Returns
    ----------
    SC_boolean :
        A boolean list for low/top or left/right single crytal region. True means the width > SC_tol.
    """
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p, str_alg, csc_tol)

    SC_boolean = True
    # w_bottom_SC = w_bottom_SC - tol_fix_reg
    # w_top_SC = w_top_SC - tol_fix_reg
    if w_bottom_SC < SC_tol:
        SC_boolean = False
    if w_top_SC < SC_tol:
        SC_boolean = False
    return SC_boolean


def add_sc(pkl_file, data_0, lat_par, rCut, non_p, tol_fix_reg, SC_tol, str_alg, csc_tol, box_bound):
    """
    Function adds single crystal region to the simulation case the GB get close to the edges

    Parameters
    ------------
    pkl_file :
    data_0 :
    lat_par :
        Lattice parameter for the crystal being simulated.
    rCut :
        Cut-off radius for computing Delaunay triangulations
    non_pbc : int
        The non-periodic direction. 0 , 1 or 2 which corresponds to
        x, y and z direction, respectively.
    tol_fix_reg : float
        The user defined tolerance for the size of rigid translation region in lammps simulation.
    SC_tol : float
        The user defined tolerance for the minimum size of single crystal region.
    str_alg : string
        The algorithm used to find the atoms in the GB and the surfaces.
        str_alg="csc" which uses centrosymmetry parameter
        str_alg="ptm" which uses polyhedral template matching
    csc_tol : float
        The tolerance for identifing the atoms in the GB and surfaces using
        the str_alg="csc"
    box_bound : np.array
        The box bound read from lammps dump file which is 9 parameters: xlo, xhi, ylo,
        yhi, zlo, zhi, xy, xz, yz

    Returns
    ----------
    uniq_atoms : numpy.ndarray
        The ID, atom type and position of the atoms in the single crystal region which will be added
        to the lammps dump file.
    """
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data_0, lat_par, non_p, str_alg, csc_tol)
    position = data_0.particles['Position'][...]
    jar = open(pkl_file, 'rb')
    gb_attr = pkl.load(jar)
    jar.close()
    sim_cell = gb_attr['cell']
    point1 = gb_attr['lpts']
    point2 = gb_attr['upts']
    sum_p1 = np.sum(point1, axis=0)
    # sum_p2 = np.sum(point2, axis=0)
    if sum_p1[non_p] < 0:
        l_pts = point1
        u_pts = point2
    else:
        l_pts = point2
        u_pts = point1

    if non_p == 0:
        d1 = sim_cell[:, 1]
        d2 = sim_cell[:, 2]
    elif non_p == 1:
        d1 = sim_cell[:, 0]
        d2 = sim_cell[:, 2]
    else:
        d1 = sim_cell[:, 0]
        d2 = sim_cell[:, 1]

    img_vec = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]])
    if GbRegion[0] < 0:
        l_pts0 = l_pts
        for i in range(8):
            new = l_pts0 + img_vec[i, 0] * d1 + img_vec[i, 1] * d2
            l_pts = np.concatenate((l_pts, new), axis=0)
    elif GbRegion[1] > 0:
        u_pts0 = u_pts
        for i in range(8):
            new = u_pts0 + img_vec[i, 0] * d1 + img_vec[i, 1] * d2
            u_pts = np.concatenate((u_pts, new), axis=0)

    if GbRegion[0] < 0:
        min_gb = GbRegion[0] - lat_par
        id2del = np.where(position[:, non_p] < min_gb)[0]
        new_atoms = np.delete(position, id2del, axis=0)
        atom_id = np.argmin(new_atoms[:, non_p])
        ref_atom = new_atoms[atom_id]
        l_pts = l_pts + ref_atom
        new_l_con = np.concatenate((l_pts, new_atoms), axis=0)
    elif GbRegion[1] > 0:
        max_gb = GbRegion[1] + lat_par
        id2del = np.where(position[:, non_p] > max_gb)[0]
        new_atoms = np.delete(position, id2del, axis=0)
        atom_id = np.argmax(new_atoms[:, non_p])
        ref_atom = new_atoms[atom_id]
        u_pts = u_pts + ref_atom
        new_l_con = np.concatenate((u_pts, new_atoms), axis=0)

    new_l_con[:, non_p] = new_l_con[:, non_p] - np.mean(GbRegion)
    new_l_con = new_l_con[np.where((new_l_con[:, non_p] < box_bound[non_p, 1]) &
                                   (new_l_con[:, non_p] > box_bound[non_p, 0]))[0]]
    uniq_atoms = np.unique(new_l_con, axis=0)
    len_all = len(uniq_atoms)
    ID = np.arange(len_all).reshape(-1, 1)
    type_atom = np.zeros(len_all)
    type_atom[0:len(new_atoms)] = 1
    type_atom[len(new_atoms):len_all] = 2
    type_atom = type_atom.reshape(-1, 1)
    uniq_atoms = np.concatenate((ID, type_atom,  uniq_atoms), axis=1)
    return uniq_atoms
