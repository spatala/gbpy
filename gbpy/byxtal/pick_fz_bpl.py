import numpy as np
import math
import os
import pickle


def pick_fz_bpl(bp_norms_go1, bp_symm_grp, symm_grp_ax, x_tol=1e-04):
    """
    Returns an array of unit boundary plane normals lying in the fundamental zone of bicrystal
    symmetry for an input array containing unit boundary plane normals in po1 reference frame.

    Parameters
    -----------
    bp_norms_go1: An array of unit boundary plane vectors in the reference frame of lower crystal 1 (po1).
    * numpy array of size (n x 3)

    bp_symm_grp: The point group symmetry of the underlying bicrystal.
    * python string with allowed values 'C_s', 'C_2h', 'D_3d', 'D_2h', 'D_4h', 'D_6h', 'D_8h' and 'O_h'

    symm_grp_ax: The principal axes of bicrystal symmetry group in orthogonal reference frame of crystal 1 (po1).
    * numpy array of size (3 x 3)
    * x_axis == symm_grp_axes[:, 0]; y_axis == symm_grp_axes[:, 1]; z_axis == symm_grp_axes[:, 2]

    x_tol : tolerance value to find points in the bicrystal fundamental zone
    * float, default value == 1e-04

    Returns
    --------
    bp_fz_norms_go1: Fundamental zone boundary plane array.
    * numpy array of size (n x 3)
    * Each row is a unit boundary plane vector in the bicrystal fundamental zone.
    * For each vector the components are expressed in orthogonal reference frame of crystal 1 (po1).

    bp_fz_stereo : Fundamental zone boundary plane array.
    * numpy array of size (n x 3)
    *  The boundary plane vectors are rotated for steregraphic projection along z_axis (symm_grp_axes[:, 2].
    """
    x_g = (symm_grp_ax[:, 0]); y_g = symm_grp_ax[:, 1]; z_g = symm_grp_ax[:, 2]

    gb_dir = os.path.dirname(os.path.realpath(__file__));
    pkl_path = gb_dir+'/data_files/';
    # gb_dir = dir_path + 'GBpy/bicrystallography'
    # pkl_path = gb_dir + '/pkl_files/'

    ### Populating symmetrically equivalent points using the symmetry operations of the bicrystal point group symmetry
    if bp_symm_grp == 'Cs':
        file_name = 'symm_mats_Cs.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'C2h':
        file_name = 'symm_mats_C2h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D3d':
        file_name = 'symm_mats_D3d.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D2h':
        file_name = 'symm_mats_D2h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D4h':
        file_name = 'symm_mats_D4h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D6h':
        file_name = 'symm_mats_D6h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'D8h':
        file_name = 'symm_mats_D8h.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)
    elif bp_symm_grp == 'Oh':
        file_name = 'symm_mats_Oh.pkl'
        file_path = pkl_path + file_name
        symm_bpn_go1 = rot_symm(symm_grp_ax, bp_norms_go1, file_path)

    ### Axes for Stereographic Projection (z-axis: along which the points are projected)
    z_g = z_g/np.linalg.norm(z_g); y_g = y_g/np.linalg.norm(y_g); x_g = x_g/np.linalg.norm(x_g)
    rot_mat = np.linalg.inv(np.column_stack((x_g, y_g, z_g)))
    ### Rotating the points
    t1_vecs = np.tensordot(rot_mat, symm_bpn_go1.transpose((2, 1, 0)), 1).transpose((2, 1, 0))
    ### Extracting the x, y and z components of the rotated points
    x = t1_vecs[:, :, 0]; y = t1_vecs[:, :, 1]; z = t1_vecs[:, :, 2];


    ### Finding the points in the bicrystal fundamental zone
    if bp_symm_grp == 'Cs':
        cond = (z > -x_tol)
    elif bp_symm_grp == 'C2h':
        cond = (z > -x_tol) & (y > -x_tol)
    elif bp_symm_grp == 'D3d':
        cond = ((z > -x_tol) & (x > -x_tol)) & ((x/math.sqrt(3) -abs(y) > -x_tol))
    elif bp_symm_grp == 'D2h':
        cond = ((z > -x_tol) & (y > -x_tol)) & (x > -x_tol)
    elif bp_symm_grp == 'D4h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/4)*x -y) > -x_tol)
    elif bp_symm_grp == 'D6h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/6)*x -y) > -x_tol)
    elif bp_symm_grp == 'D8h':
        cond = ((z > -x_tol) & (y > -x_tol)) & ((math.tan(np.pi/8)*x -y) > -x_tol)
    elif bp_symm_grp == 'Oh':
        cond = (((z - x) > -x_tol) & ((x - y) > -x_tol)) & (y > -x_tol)

    # return np.where(cond)[0]

    ### Selecting the first point in case of multiple symmetrically equivalent points in the FZ ###
    num = np.shape(cond)[1]
    bpn_fz = []
    for ct1 in range(num):
        t2_vecs = t1_vecs[:, ct1, :]
        t3_vecs = t2_vecs[cond[:, ct1]][0]
        bpn_fz.append(t3_vecs)
    bp_fz_stereo = np.array(bpn_fz)

    ### Rotating the points back to the orthogonal frame of lower crystal 1 (po1).
    bp_fz_norms_go1 = np.dot(np.linalg.inv(rot_mat), bp_fz_stereo.transpose()).transpose()

    return bp_fz_norms_go1, bp_fz_stereo


def rot_symm(symm_grp_ax, bp_norms_go1, file_path):
    """
    Returns the symmetrically equivalent boundary plane normals using the symmetry operations of bicrystal point
    group symmetry.

    Parameters
    ----------
    symm_grp_ax: The principal axes of bicrystal symmetry group in orthogonal reference frame of crystal 1 (po1).
    * numpy array of size (3 x 3)
    * x_axis == symm_grp_axes[:, 0]; y_axis == symm_grp_axes[:, 1]; z_axis == symm_grp_axes[:, 2]

    bp_norms_go1: normalized boundary plane normals in the po1 reference frame
    * numpy array of size (n x 3)

    file_path: path to the relevant symmetry operations containing pickle file
    * python string

    Returns
    -------
    symm_bpn_go1: symmetrically equivalent boundary plane normals in the po1 reference frame
    * numpy array of size (m x n x 3); m == order of bicrystal point group symmetry group

    Notes
    ------

    """
    bpn_rot = np.dot(np.linalg.inv(symm_grp_ax), bp_norms_go1.transpose()).transpose()
    symm_mat = pickle.load(open(file_path, 'rb'), encoding='latin1')
    ### np.dot returns the sum product of the last axis of the first matrix with the second to last axis of the second
    ### advisable to use np.tensordot instead to avoid confusion !!
    symm_bpn_rot_gop1 = np.tensordot(symm_mat, bpn_rot.transpose(), 1).transpose((1, 2, 0))
    symm_bpn_go1 = np.tensordot(symm_grp_ax, symm_bpn_rot_gop1, 1).transpose(2, 1, 0)
    return symm_bpn_go1
