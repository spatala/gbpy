import numpy as np
import math


def disorient_symm_props(mis_quat_fz, lat_pt_grp, x_tol=1e-04):
    """
    Returns the point group and the principle axes for fundamental zone of underlying bicrystal symmetry.
    This method takes only one value for lat_pt_grp == 'O_h', i.e. the function is written for bcc and fcc
    crystals only.
    Based on the location of the mis_quat_fz in the quaternion hypersphere (4-D), the point group symmetry
    of the bicrystal is determined.

    Parameters
    -----------
    mis_quat_fz: numpy.array
        A quaternion array with size (5 x 1)
        The quaternion is created from the misorientation of the sigma value for the grain boundary.
        The misorientation is defined in the orthogonal reference frame of lower crystal 1 (po1).
        lat_pt_grp: The point group symmetry of the crystal.
        string with allowed value 'Oh'
    x_tol: float
        Tolerance value to check various conditions in the function, default value==1e-04

    Returns
    --------
    bp_symm_grp: str
        The point group symmetry of bicrystal. python string with allowed
        values 'Cs', 'C2h', 'D3d', 'D2h', 'D4h', 'D6h', 'D8h' and 'Oh'
    x_g: int
        First principle axes for the fundamental zone of the bicrystal.
    y_g: int
        Second principle axes for the fundamental zone of the bicrystal.
    z_g: int
        Third principle axes for the fundamental zone of the bicrystal.

    """
    q0 = mis_quat_fz[0][0]
    q1 = mis_quat_fz[1][0]
    q2 = mis_quat_fz[2][0]
    q3 = mis_quat_fz[3][0]

    if lat_pt_grp == 'Oh':
        k = math.sqrt(2) - 1
        k1 = 1.0 / math.sqrt(1 + 2*k*k)

        cond_0 = abs(q0 - 1) <= x_tol
        pt_o = cond_0
        if pt_o:
            z_g = np.array([0, 0, 1])
            x_g = np.array([1, 0, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'Oh'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q0 - math.cos(np.pi/8)) <= x_tol
        cond_1 = abs(q1) <= x_tol
        cond_2 = abs(q2) <= x_tol
        cond_3 = abs(q3 - math.sin(np.pi/8)) <= x_tol

        pt_a = cond_0 and cond_1 and cond_2 and cond_3
        if pt_a:
            z_g = np.array([1, 0, 0])
            x_g = np.array([0, -q3, q0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D8h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q0 - math.sqrt(3)/2) <= x_tol
        cond_1 = abs(q1 - 1/(2*math.sqrt(3))) <= x_tol
        cond_2 = abs(q2 - 1/(2*math.sqrt(3))) <= x_tol
        cond_3 = abs(q3 - 1/(2*math.sqrt(3))) <= x_tol

        pt_e = cond_0 and cond_1 and cond_2 and cond_3
        if pt_e:
            z_g = np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
            x_g = np.array([2/math.sqrt(3), -1/math.sqrt(3), -1/math.sqrt(3)])/math.sqrt(2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D6h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q0 - 1/(k*2*math.sqrt(2))) <= x_tol
        cond_1 = abs(q1 - 1/(2*math.sqrt(2))) <= x_tol
        cond_2 = abs(q2 - 1/(2*math.sqrt(2))) <= x_tol
        cond_3 = abs(q3 - k/(2*math.sqrt(2))) <= x_tol

        pt_c = cond_0 and cond_1 and cond_2 and cond_3
        if pt_c:
            z_g = np.array([0, -1/math.sqrt(2), 1/math.sqrt(2)])
            x_g = np.array([1, 0, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D4h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q2 - q3) <= x_tol
        cond_1 = abs(q3 - 0) <= x_tol
        cond_2 = abs(q1 - math.sin(np.pi/8)) >= x_tol
        cond_3 = abs(q0 - 1) >= x_tol

        line_oa = cond_0 and cond_1 and cond_2 and cond_3
        if line_oa:
            z_g = np.array([1, 0, 0])
            x_g = np.array([0, -q1, q0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D4h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 - q2) <= x_tol
        cond_1 = abs(q2 - q3) <= x_tol
        cond_2 = abs(q3 - 1/(2*math.sqrt(3))) >= x_tol
        cond_3 = abs(q0 - 1) >= x_tol

        line_oe = cond_0 and cond_1 and cond_2 and cond_3
        if line_oe:
            z_g = np.array([1/math.sqrt(3), 1/math.sqrt(3), 1/math.sqrt(3)])
            x_g = np.array([q0+q1, q1-q0, -2*q1])/math.sqrt(2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D3d'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 - q2) <= x_tol
        cond_1 = abs(q3 - 0) <= x_tol
        cond_2 = abs(q2 - math.sin(np.pi/4)/math.sqrt(2)) >= x_tol
        cond_3 = abs(q0 - 1) >= x_tol

        line_ob = cond_0 and cond_1 and cond_2 and cond_3
        if line_ob:
            z_g = np.array([1/math.sqrt(2), 1/math.sqrt(2), 0])
            x_g = np.array([q1, -q1, q0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 - q2) <= x_tol
        cond_1 = abs(2*q1 + q3 - q0) <= x_tol
        cond_2 = abs(q3 - 1/(2*math.sqrt(2))) >= x_tol

        line_ce = cond_0 and cond_1 and cond_2
        if line_ce:
            z_g = np.array([0, -1/math.sqrt(2), 1/math.sqrt(2)])
            x_g = np.array([2*q1, q1+q3, q1+q3])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q2 - q3) <= x_tol
        cond_1 = abs(q1 + 2*q2 - q0) <= x_tol
        cond_2 = abs(q3 - 1/(2*math.sqrt(3))) >= x_tol

        line_ed = cond_0 and cond_1 and cond_2
        if line_ed:
            z_g = np.array([1/math.sqrt(2), 0, -1/math.sqrt(2)])
            x_g = np.array([q1+q2, 2*q2, q1+q2])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q0 - k1) <= x_tol
        cond_1 = abs(q1 - k*k1) <= x_tol
        cond_2 = abs(q2 - k*k1) <= x_tol
        cond_3 = abs(q3 - 0) <= x_tol

        pt_b = cond_0 and cond_1 and cond_2 and cond_3
        if pt_b:
            z_g = np.array([1/math.sqrt(2), 1/math.sqrt(2), 0])
            x_g = np.array([q1, -q1, q0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 - k*q0) <= x_tol
        cond_1 = abs(q3 - k*q2) <= x_tol
        cond_2 = abs(q1 - q2) >= x_tol
        cond_3 = abs(q1 - 0) >= x_tol

        line_ac = cond_0 and cond_1 and cond_2 and cond_3
        if line_ac:
            z_g = np.array([0, 1/math.sqrt(2), 1/math.sqrt(2)])
            x_g = np.array([1, 0, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q3 - 0) <= x_tol
        cond_1 = abs(q1 - q2) >= x_tol
        cond_2 = abs(q2 - 0) >= x_tol

        surf_oab = cond_0 and cond_1 and cond_2
        if surf_oab:
            z_g = np.array([q2, -q1, q0])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 - q2) <= x_tol
        cond_1 = abs(q3 - 0) >= x_tol
        cond_2 = abs(q3 - q1) >= x_tol
        cond_3 = abs(2*q1 + q3 - q0) >= x_tol

        surf_obce = cond_0 and cond_1 and cond_2 and cond_3
        if surf_obce:
            z_g = np.array([(q0+q3)/math.sqrt(2), (q3-q0)/math.sqrt(2), - math.sqrt(2)*q1])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q2 - q3) <= x_tol
        cond_1 = abs(q2 - 0) >= x_tol
        cond_2 = abs(q1 - q2) >= x_tol
        cond_3 = abs(q1 + 2*q2 - q0) >= x_tol

        surf_oade = cond_0 and cond_1 and cond_2 and cond_3
        if surf_oade:
            z_g = np.array([-math.sqrt(2)*q2, (q0+q1)/math.sqrt(2), (q1-q0)/math.sqrt(2)])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        cond_0 = abs(q1 + q2 + q3 - q0) <= x_tol
        cond_1 = abs(q1 - q2) >= x_tol
        cond_2 = abs(q2 - q3) >= x_tol

        surf_cde = cond_0 and cond_1 and cond_2
        if surf_cde:
            z_g = np.array([q0-q3, q0-q1, q0-q2])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        z_g = np.array([q1, q2, q3])
        x_g = np.array([q2, -q1, 0])
        y_g = np.cross(z_g, x_g)
        bp_symm_grp = 'Cs'
        return x_g, y_g, z_g, bp_symm_grp


    if lat_pt_grp == 'D6h':
        k = 2 - np.sqrt(3)
        ########################### ID (1) from tbale 17############################
        cond_0 = abs(q1 - np.sqrt(3) * q2) <= x_tol
        cond_1 = abs(q2) >= x_tol
        cond_2 = abs(q3) >= x_tol
        s_ocde = cond_0 and cond_1 and cond_2 
        if s_ocde:
            z_g = np.array([(q0 + np.sqrt(3) * q3)/2, (q3 - np.sqrt(3) * q0) / 2, -2 * q1 / np.sqrt(3)])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (2) from tbale 17############################
        cond_0 = abs(q2) <= x_tol
        cond_1 = abs(q1) >= x_tol
        cond_2 = abs(q3) >= x_tol
        s_oafe = cond_0 and cond_1 and cond_2
        if s_oafe:
            z_g = np.array([-q3, q0, q1])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (3) from tbale 17############################
        cond_0 = abs(q3) <= x_tol
        cond_1 = abs(q1) >= x_tol
        cond_2 = abs(q2) >= x_tol
        s_ocba = cond_0 and cond_1 and cond_2
        if s_ocba:
            z_g = np.array([q2, -q1, q0])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (4) from tbale 17############################
        cond_0 = abs(q2 - q3) <= x_tol
        cond_1 = abs(q2) <= x_tol
        cond_2 = abs(q0 - 1) >= x_tol
        cond_3 = abs(q1 - math.sin(np.pi/4)) >= x_tol

        line_oa = cond_0 and cond_1 and cond_2 and cond_3
        if line_oa:
            z_g = np.array([1, 0, 0])
            x_g = np.array([0, q0, q1])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (5) from tbale 17############################
        cond_0 = abs(q1 - np.sqrt(3)*q2) <= x_tol
        cond_1 = abs(q3) <= x_tol
        cond_2 = abs(q2 - .5*math.sin(np.pi/4)) >= x_tol
        cond_3 = abs(q0 - 1) >= x_tol

        line_oc = cond_0 and cond_1 and cond_2 and cond_3
        if line_oc:
            z_g = np.array([np.sqrt(3)/2, .5, 0])
            x_g = np.array([q1/np.sqrt(3), -q1, -q0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (6) from tbale 17############################
        cond_0 = abs(q1 - q2) <= x_tol
        cond_1 = abs(q2) <= x_tol
        cond_2 = abs(q3 - math.sin(np.pi/12)) >= x_tol
        cond_3 = abs(q0 - 1) >= x_tol

        line_oe = cond_0 and cond_1 and cond_2 and cond_3
        if line_oe:
            z_g = np.array([0, 0, 1])
            x_g = np.array([q0, q3, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D6h'
            return x_g, y_g, z_g, bp_symm_grp

        ###########ADD number 7 in tbale 17###############
        cond_0 = abs(q1 - q0) <= x_tol
        cond_1 = abs(q2 - q3) <= x_tol
        cond_2 = abs(q2) >= x_tol
        cond_3 = abs(q2 - k/np.sqrt(8*k)) >= x_tol
        cond_4 = abs(np.sqrt(3)*q1 + q2 - 2*q0) <= x_tol
        cond_5 = abs(2*q1 - q3 - np.sqrt(3)*q0) <= x_tol
        line_ag = cond_0 and cond_1 and cond_2 and cond_3 and cond_4 and cond_5
        if line_ag:
            z_g = np.array([0, 0, 1])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp

        ###########ADD number 8 in tbale 17###############
        cond_0 = abs(q3) >= x_tol
        cond_1 = abs(q3 - k/np.sqrt(8*k)) >= x_tol
        line_cg = cond_0 and cond_1
        if line_cg:
            z_g = np.array([-.5, np.sqrt(3)/2, 0])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp
        ###########ADD number 9 in tbale 17###############
        cond_0 = abs(q3 - k*q0) <= x_tol
        cond_1 = abs(q2 - k*q1) <= x_tol
        cond_2 = abs(q1) >= x_tol
        cond_3 = abs(q1 - 1/np.sqrt(8*k)) >= x_tol

        line_eg = cond_0 and cond_1 and cond_2 and cond_3
        if line_eg:
            z_g = np.array([np.sqrt(3)/2, 1/2, 0])
            x_g = np.array([q1, q2, q3])/math.sqrt(1-q0**2)
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'C2h'
            return x_g, y_g, z_g, bp_symm_grp
        ###########ADD number 10 in tbale 17###############      
        cond_0 = abs(q0 - math.cos(np.pi/4)) <= x_tol
        cond_1 = abs(q1 - math.sin(np.pi/4)) <= x_tol
        cond_2 = abs(q2) <= x_tol
        cond_3 = abs(q3) <= x_tol

        p_a = cond_0 and cond_1 and cond_2 and cond_3
        if p_a:
            z_g = np.array([1, 0, 0])
            x_g = np.array([0, 1, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D4h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (11) from tbale 17############################
        cond_0 = abs(q0 - math.cos(np.pi/4)) <= x_tol
        cond_1 = abs(q1 - np.sqrt(3)*math.sin(np.pi/4)/4) <= x_tol
        cond_2 = abs(q2 - .5*math.sin(np.pi/4)) <= x_tol
        cond_3 = abs(q3) <= x_tol

        p_c = cond_0 and cond_1 and cond_2 and cond_3
        if p_c:
            z_g = np.array([np.sqrt(3)/2, .5, 0])
            x_g = np.array([0, 0, 1])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D4h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (12) from tbale 17############################
        cond_0 = abs(q0 - math.cos(np.pi/12)) <= x_tol
        cond_1 = abs(q1) <= x_tol
        cond_2 = abs(q2) <= x_tol
        cond_3 = abs(q3 - math.sin(np.pi/12)) <= x_tol

        line_ed = cond_0 and cond_1 and cond_2 and cond_3
        if line_ed:
            z_g = np.array([0, 0, 1])
            x_g = np.array([q0, q3, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D12h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (13) from tbale 17############################
        cond_0 = abs(q0 - q1) <= x_tol
        cond_1 = abs(q1 - 1/(np.sqrt(8*k))) <= x_tol
        cond_2 = abs(q2 - q3) <= x_tol
        cond_3 = abs(q3 - k/np.sqrt(8*k)) <= x_tol
        p_g = cond_0 and cond_1 and cond_2 and cond_3
        if p_g:
            z_g = np.array([0, 0, 1])
            x_g = np.array([np.sqrt(3)/2, .5, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D2h'
            return x_g, y_g, z_g, bp_symm_grp
        ########################### ID (14) from tbale 17############################
        cond_0 = abs(q0 - 1) <= x_tol
        p_o = cond_0 
        if p_o:
            z_g = np.array([0, 0, 1])
            x_g = np.array([1, 0, 0])
            z_g = z_g/np.linalg.norm(z_g)
            x_g = x_g/np.linalg.norm(x_g)
            y_g = np.cross(z_g, x_g)
            bp_symm_grp = 'D6h'
            return x_g, y_g, z_g, bp_symm_grp