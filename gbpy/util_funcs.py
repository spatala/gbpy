import numpy as np
import ovito.io as oio
import ovito.modifiers as ovm
from itertools import islice
import pad_dump_file as pdf
import bisect


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
    filename0 : string
        The lammps dump file
    CohEng	: float
        The cohesive energy

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
    # min_rad = np.min(cc_rad)
    # max_rad = np.max(cc_rad)
    # rad_norm = (cc_rad - min_rad) / (max_rad - min_rad)
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
    path2dump :
        The path to dump the lammps dump file
     cc_coors1 :
        The coordinates of the circum-center of the tetrahedrons.

    atom_id :
        The atom ID of the inserted atom which is the ID of last atom + 1

    Return
    ----------------

    """
    lines = open(filename0, 'r').readlines()
    lines[1] = '0\n'
    lines[3] = str(int(lines[3]) + 1)
    new_line = str(atom_id) + ' 1 ' + str(cc_coors1[0]) + ' ' + str(cc_coors1[1]) + ' '\
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
    path2dump :
        The path to dump the lammps dump file
    ID2change :
        The ID of the atom which will be removed
    var :
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
    non_p :
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

    Returns
    ----------
    SC_boolean :
        A boolean list for low/top or left/right single crytal region. True means the width > SC_tol.
    """
    GbRegion, GbIndex, GbWidth, w_bottom_SC, w_top_SC = pdf.GB_finder(data, lat_par, non_p, str_alg, csc_tol)

    SC_boolean = [True, True]
    # w_bottom_SC = w_bottom_SC - tol_fix_reg
    # w_top_SC = w_top_SC - tol_fix_reg
    if w_bottom_SC < SC_tol:
        SC_boolean[0] = False
    if w_top_SC < SC_tol:
        SC_boolean[1] = False
    return SC_boolean
