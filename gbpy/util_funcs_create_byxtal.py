## Import modules
import math as mt;
import numpy as np;
import byxtal.find_csl_dsc as fcd;
import byxtal.integer_manipulations as iman;
import byxtal.bp_basis as bpb;
from sympy.matrices import Matrix, eye, zeros;
import byxtal.pick_fz_bpl as pfb;

import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm
from ovito.data import CutoffNeighborFinder

#--------------------------------------------------------------------------------------------------

def find_int_solns(a_vec, b_vec):
    """
    Given two basis vectors (a_vec and b_vec) in the primitive basis,
    find the third basis vector (c_vec) such that the matrix
    [a_vec, b_vec, c_vec] is a valid basis.
    All the components of the vectors are integers and
    the determinant of the matrix must be equal to **1**.

    Parameters
    -----------------
    a_vec: numpy.array
        The first basis vector. Must be an integer array.
    b_vec: numpy.array
        The second basis vector. Must be an integer array.

    Returns
    ------------
    l_p2_p1: numpy.array, (3X3, must be an integer array)
        A 3x3 numpy array of integers that forms the new basis for the lattice.

    """
    if isinstance(a_vec, Matrix):
        a_vec = np.array(a_vec, dtype='int64');
    if isinstance(b_vec, Matrix):
        b_vec = np.array(b_vec, dtype='int64');

    a1 = a_vec[0];a2 = a_vec[1];a3 = a_vec[2];
    b1 = b_vec[0];b2 = b_vec[1];b3 = b_vec[2];
    a = a2*b3 - a3*b2; b = -(a1*b3 - a3*b1); c = a1*b2 - a2*b1;
    d = 1;
    a = int(a); b = int(b); c = int(c); d = int(d);

    ####
    p = mt.gcd(a,b);
    if p==0:
        if c == 1:
            y1 = 0; y2 = 0; y3 = 1;

            # l_p2_p1 = np.transpose(np.vstack((a_vec, b_vec, np.array([y1,y2,y3]))));
            l_p2_p1 = (np.hstack((a_vec, b_vec, np.array([[y1],[y2],[y3]]))));
            det1 = np.linalg.det(l_p2_p1);
            if ((np.abs(det1)-1) > 1e-10):
                raise Exception('Error with Diophantine solution')
            else:
                if det1 == -1:
                    l_p2_p1[:,2] = -l_p2_p1[:,2];
        else:
            raise Exception('Error with boundary-plane indices')
    else:
        a1 = int(a/p); b1 = int(b/p);
        # Let u0 and v0 any solution of a'u + b'v = c
        int_soln1 = bpb.lbi_dioph_soln(a1, b1, c)
        u0 = int(int_soln1[0]); v0 = int(int_soln1[1]);
        # z0, t0 any solution of cz + pt = d
        int_soln2 = bpb.lbi_dioph_soln(c, p, d)
        z0 = int(int_soln2[0]); t0 = int(int_soln2[1]);
        # x0, y0 any solution of a'x + b'y = t0
        int_soln3 = bpb.lbi_dioph_soln(a1, b1, t0)
        x0 = int(int_soln3[0]); y0 = int(int_soln3[1]);
        # The general solution of ax + by + cz = d is :
        # x = x0 + b'k - u0m
        # y = y0 - a'k - v0m
        # z = z0 + pm with k and m any integer in Z
        tn1 = 10;
        ival = np.arange(-(tn1),tn1+1)
        k1, m1 = np.meshgrid(ival, ival)
        k1 = k1.flatten(); m1 = m1.flatten();
        x = (x0 + b1*k1 - u0*m1);
        y = (y0 - a1*k1 - v0*m1);
        z = (z0 + p*m1);

        l2_val = x**2 + y**2 + z**2;
        ind1 = np.where(l2_val == np.min(l2_val))[0][0];
        y1 = x[ind1]; y2 = y[ind1]; y3 = z[ind1];

        # l_p2_p1 = np.transpose(np.vstack((a_vec, b_vec, np.array([y1,y2,y3]))));
        l_p2_p1 = (np.hstack((a_vec, b_vec, np.array([[y1],[y2],[y3]]))));
        det1 = np.linalg.det(l_p2_p1);
        if (np.abs(det1-1) > (1e-10*np.max(np.abs(l_p2_p1)))):
            raise Exception('Error with Diophantine solution')
        else:
            if det1 == -1:
                l_p2_p1[:,2] = -l_p2_p1[:,2];

    return Matrix((l_p2_p1).astype(int));

def compute_rCut(l2d_bp_po):
    """
    Given two vectors in the interface plane, compute the
    maximum of the norms of the two vectors.

    Parameters
    -----------------
    l2d_bpb_po: numpy.array
        The two vectors, expressed in the **po** reference frame,
        that define the two-dimensional box vectors of the interface.

    Returns
    ------------
    rCut: float
        The cut-off radius for replicating the lattice basis.
    """
    bv1 = l2d_bp_po[:,0]; bv2 = l2d_bp_po[:,1];
    l1 = bv1.norm(); l2 = bv2.norm();
    l3 = (bv1+bv2).norm();
    rCut = np.max([l1,l2,l3])
    return rCut;

def compute_orientation(l2d_bp_po):
    """
    Find the orientation of the lattice **l_po1_go**, such that the
    vectors in the **p1** lattice given by **l2d_bp_po** to line up
    with x-axis in the xy-plane.
    """
    bv1 = l2d_bp_po[:,0]; bv2 = l2d_bp_po[:,1];
    l1 = bv1.norm(); l2 = bv2.norm();
    ### Orientation
    l1_uvec = bv1/l1; l2_uvec = bv2/l2;
    x_vec = Matrix(l1_uvec); y1_vec = Matrix(l2_uvec);
    z_vec = x_vec.cross(y1_vec);
    z_vec = z_vec/z_vec.norm();
    y_vec = z_vec.cross(x_vec);
    # l_po1_go = Matrix((np.vstack((x_vec, y_vec, z_vec))).transpose());
    l_po1_go = (x_vec.row_join(y_vec)).row_join(z_vec);
    return l_po1_go.inv();

def compute_hkl_p(l2d_bp_po, l_p_po):
    """
    Find the **(hkl)** indices of the plane defined the vectors
    in the matrix **l2d_bp_po**.

    """
    avec_po = l2d_bp_po[:,0]; bvec_po = l2d_bp_po[:,1];
    # nvec = np.cross(avec_po, bvec_po);
    nvec = avec_po.cross(bvec_po);
    # nu_vec = nvec/np.linalg.norm(nvec);
    nu_vec = nvec/nvec.norm();
    l_rp_po = fcd.reciprocal_mat(l_p_po);
    l_po_rp = l_rp_po.inv();
    nu_vec_rp = l_po_rp*nu_vec;
    return iman.int_finder(nu_vec_rp);

def num_rep_2d(xvec, yvec, rCut):
    """
    Find the number of replications necessary such that the
    2D-circle of radius r_cut at the center of the primitive-cell
    lies completely inside the super-cell.
    """
    xvec1 = xvec.col_join(Matrix([0]));
    yvec1 = yvec.col_join(Matrix([0]));

    c_vec_norm = (xvec1.cross(yvec1)).norm();
    d_y = c_vec_norm/(yvec.norm());
    d_x = c_vec_norm/(xvec.norm());
    m_x = np.ceil(rCut/d_y);
    m_y = np.ceil(rCut/d_x);

    return [int(m_x), int(m_y)];


def replicate_pts(l_bpb_po, rCut):
    """
    Replicate the basis, enough times, such that the 2D-circle
    of radius r_cut is completely inside the replicated set.
    """
    bx = l_bpb_po[:,0]; by = l_bpb_po[:,1];
    mx, my = num_rep_2d(bx, by, rCut);

    mx1 = np.arange(-mx,mx+1); my1 = np.arange(-my,my+1);
    mx2, my2 = np.meshgrid(mx1, my1);
    mx3 = mx2.flatten(); my3 = my2.flatten();

    num1 = np.size(mx3);
    twoD_pts = np.zeros((num1,2));
    bx1 = bx.reshape(1,2);
    by1 = by.reshape(1,2);
    for ct1 in range(num1):
        mx_val = mx3[ct1]; my_val = my3[ct1];
        twoD_pts[ct1, :] = (mx_val*bx1 + my_val*by1)
    return twoD_pts;

def change_basis(twoD_pts, l_bpb_po):
    """
    Express the points in **twoD_pts** array in the reference frame
    of **l_bpb_po**.
    """
    mat1 = (l_bpb_po).inv();
    return np.array((mat1*(Matrix(twoD_pts.transpose()))).transpose(), dtype='double');

def cut_box_pts(twoD_pts, tol=1e-8):
    """
    Remove all the points, in 2D, that lie outside the 2D box.
    """
    tx1 = twoD_pts[:,0]; ty1 = twoD_pts[:,1];
    cond1 = (tx1 >= 0-tol); cond2 = (tx1 <= 1+tol);
    cond3 = (ty1 >= 0-tol); cond4 = (ty1 <= 1+tol);
    tind1 = np.where(cond1 & cond2 & cond3 & cond4)[0];
    return twoD_pts[tind1,:];

def knnsearch_v1(X, Y):
    """
    Given two set of points **X** and **Y**, find the nearest neighbor in
    **X** for each query point in **Y** and return the indices of
    the nearest neighbors in Idx, a column vector. Idx has the same
    number of rows as Y. Additionally returns the column vector **D** that
    contains the nearest-neighbor distances.
    """
    num_x = np.shape(X)[0]; num_y = np.shape(Y)[0];
    dval = np.zeros((num_y,));
    Idx = np.zeros((num_y,), dtype='int');
    for ct1 in range(num_y):
        tpt2 = Y[ct1,:];
        diff1 = X - np.tile(tpt2, (num_x,1));
        dval1 = np.sqrt(np.sum(diff1**2,axis=1));
        dval[ct1] = np.min(dval1);
        Id1 = np.where(dval1 == np.min(dval1))[0];
        Idx[ct1] = int(Id1[0]);
    return Idx, dval;

def overlap_inds(pts, tx_vec, d_tol):
    """
    Translate pts by the tx_vec vector. Return the indices
    of points in pts that would overlap with the translated
    set of points.
    """
    num1 = np.shape(pts)[0];
    tpts2 = pts+np.tile(tx_vec.transpose(), (num1,1));
    Idx, D = knnsearch_v1(tpts2, pts);
    return Idx[np.where(D < d_tol)[0]];

def find_bound_inds(twoD_pts, tol1):
    """
    Given set of **twoD_pts** in fractional coordinates,
    provide the indices of the points that are close to the
    boundaries of the box.
    """
    cond1 = (np.abs(twoD_pts[:,0]) < tol1);
    cond2 = (np.abs(twoD_pts[:,0]-1) < tol1);
    cond3 = (np.abs(twoD_pts[:,1]) < tol1);
    cond4 = (np.abs(twoD_pts[:,1]-1) < tol1);
    ind1 = np.where(cond1 | cond2 | cond3 | cond4)[0];

    ind2 = np.where(~(cond1 | cond2 | cond3 | cond4))[0];
    return ind1, ind2;

def remove_2d_overlaps(l_bp_po, ind1, ind2, pts1):
    """
    Given the two vectors of that define the box and the set of points (pts1)
    inside the box, remove the set of points that will overlap under
    periodic boundary conditions.
    """
    x0 = np.arange(-1,2);
    x1, y1 = np.meshgrid(x0, x0);
    x2 = x1.flatten(); y2 = y1.flatten();
    mg1 = (np.vstack((x2,y2))).transpose();
    tind1 = np.where((mg1[:,0] == 0) & (mg1[:,1] == 0))[0][0];
    mg1 = np.delete(mg1, tind1, axis=0);

    #### Translate pts1[ind1,:] and remove overlaps
    tx_vec = np.array(l_bp_po[:,0], dtype='double');
    ty_vec = np.array(l_bp_po[:,1], dtype='double');
    ####
    num_trans = np.shape(mg1)[0];
    for ct1 in range(num_trans):
        # print(ct1)
        mx = mg1[ct1,0]; my = mg1[ct1,1];
        tpts1 = pts1[ind1,:]
        i2 = overlap_inds(tpts1, mx*tx_vec + my*ty_vec, 0.1);
        tpts1 = np.delete(tpts1, i2, axis=0); ind1 = np.delete(ind1, i2);

    twoD_pts = np.vstack((pts1[ind1,:], pts1[ind2,:]));
    return twoD_pts

def create_twoD_slab(l_bp_po, l_p_po):
    """
    Given the two vectors of the 2D simulation box in the plane, **l_bp_po**,
    and the basis vectors of the lattice, **l_p_po**, create the 2D slab of
    atoms (that satisfies the periodic boundary conditions). The plane is
    oriented such that the two vectors in **l_bp_po** are in the X-Y plane.
    """
    rCut = compute_rCut(l_bp_po);
    ################################################################################
    ############ Lower crystal orientation
    ### Find the lcryst vectors along a-vec, b-vec
    l_po1_go = compute_orientation(l_bp_po);
    l2D_bpbSig_po1 = l_po1_go*l_bp_po;

    hkl_p1 = compute_hkl_p(l_bp_po, l_p_po);
    l_bpb_p1 = bpb.bp_basis(hkl_p1);
    l_bpb_p1 = Matrix(l_bpb_p1.astype(int));
    l_bpb_po1 = l_p_po*l_bpb_p1;

    l2D_bpb_po1 = l_po1_go*l_bpb_po1;
    twoD_mat = l2D_bpb_po1[:2,:];

    twoDSig_mat = l2D_bpbSig_po1[:2,:];
    twoD_pts = replicate_pts(twoD_mat, rCut);

    twoD_pts = remove_periodic_overlaps(twoD_pts, twoDSig_mat)
    # import matplotlib.pyplot as plt; fig1 = plt.figure();
    # import plotting_routines as pr; pr.plot_2d_box_box(twoD_mat, twoDSig_mat);
    # pts = np.copy(twoD_pts); x1=pts[:,0];y1=pts[:,1]; plt.scatter(x1,y1); plt.show();


    # twoD_pts_sc1 = change_basis(twoD_pts, twoDSig_mat);
    # twoD_pts_sc1 = cut_box_pts(twoD_pts_sc1, 1e-8);
    # tol1 = 1e-1;
    # ind1, ind2 = find_bound_inds(twoD_pts_sc1, tol1);
    # twoDSig_mat_inv = twoDSig_mat.inv();
    # pts1 = change_basis(twoD_pts_sc1, twoDSig_mat_inv);
    # twoD_pts = remove_2d_overlaps(twoDSig_mat, ind1, ind2, pts1);
    # ################################################################################

    # import matplotlib.pyplot as plt; fig2 = plt.figure();
    # import plotting_routines as pr; pr.plot_2d_box_box(twoD_mat, twoDSig_mat);
    # pts = np.copy(twoD_pts); x1=pts[:,0];y1=pts[:,1]; plt.scatter(x1,y1); plt.show();

    return l_bpb_p1, l2D_bpbSig_po1, twoD_pts;

def create_threeD_slab(zCut, tz_vec, l_bp_po, twoD_pts):
    """
    Given the two vectors of the 2D simulation box in the plane, **l_bp_po**,
    and the basis vectors of the lattice, **l_p_po**, create the 2D slab of
    atoms (that satisfies the periodic boundary conditions). The plane is
    oriented such that the two vectors in **l_bp_po** are in the X-Y plane.
    """

    if isinstance(tz_vec, Matrix):
        tz_vec = np.array(tz_vec, dtype='double');
        tz_vec = tz_vec.reshape(3,)
    if isinstance(l_bp_po, Matrix):
        l_bp_po = np.array(l_bp_po, dtype='double');

    ################################################################################
    ## Translate 2D points in the Z-direction with zCut
    num_rep = np.abs(int(np.ceil(zCut/np.dot(tz_vec,np.array([0,0,1])))));
    num_2d = np.shape(twoD_pts)[0];
    num_3d_pts = int((2*num_rep+1)*num_2d); # num_3d_pts = int(num_rep*num_2d);
    threeD_pts = np.zeros((num_3d_pts,3));

    twoD_pts1 = np.hstack((twoD_pts, np.zeros((num_2d,1))));

    for ct1 in np.arange(-num_rep, num_rep+1):
        ct2 = ct1 + num_rep;
        ind_st = (ct2)*num_2d;
        ind_stop = ind_st + num_2d;
        trans_vec = tz_vec*ct1;
        threeD_pts[ind_st:ind_stop, :] = twoD_pts1 + np.tile(trans_vec, (num_2d,1));

    ### Simulation Cell Box
    ### Following Ovito's convention
    sim_cell = np.zeros((3,4)); sim_avec = l_bp_po[:,0]; sim_bvec = l_bp_po[:,1];
    ### Change this with inter-planar spacing
    sim_cvec = np.array([0,0,2*zCut]); # sim_cvec = np.array([0,0,zCut]);
    sim_orig = np.array([0,0,-zCut]); # sim_orig = np.array([0,0,0]);

    sim_cell[:,0] = sim_avec; sim_cell[:,1] = sim_bvec;
    sim_cell[:,2] = sim_cvec; sim_cell[:,3] = sim_orig;

    box_vecs = sim_cell[:,0:3];
    tpts1 = np.dot(np.linalg.inv(box_vecs), threeD_pts.transpose()).transpose();

    threeD_pts1 = wrap_cc(sim_cell, threeD_pts);

    # import matplotlib.pyplot as plt; fig1 = plt.figure();
    # from mpl_toolkits.mplot3d import Axes3D; ax = fig1.add_subplot(111, projection='3d');
    # pts = np.copy(tpts1); x1=pts[:,0];y1=pts[:,1];z1=pts[:,2]; ax.scatter(x1,y1,z1); plt.show();

    # import plotting_routines as pr;
    # pr.plot_2d_pts_box(twoD_pts, l_bp_po[0:2,:]);
    # import matplotlib.pyplot as plt;
    # pts = np.copy(twoD_pts); x1=pts[:,0];y1=pts[:,1]; plt.scatter(x1,y1);
    # plt.show();

    for ct1 in range(2):
        tpts_x = tpts1[:,ct1];
        ## modf:
        ## Return the fractional and integral parts of an array, element-wise.
        y1, y2 = np.modf(tpts_x);
        tpts_x = tpts_x - y2;
        ind1 = np.where(tpts_x < 0)[0];
        tpts_x[ind1] = tpts_x[ind1] + 1;
        tpts1[:,ct1] = tpts_x;

    threeD_pts = np.dot(box_vecs, tpts1.transpose()).transpose();

    # import matplotlib.pyplot as plt; fig1 = plt.figure();
    # import plotting_routines as pr; pr.plot_3d_pts_box(fig1, threeD_pts, sim_cell[:,0:3], sim_orig); plt.show();


    return threeD_pts, sim_cell;


def create_half_cryst(l_csl_p1, l_bp_CSLp, l_p_po, cryst_typ, zCut):
    """

    """
    l_csl_po1 = l_p_po*l_csl_p1;
    l_bp_po1 = l_csl_po1*l_bp_CSLp;
    l_bpb_p1, l2D_bp_po1, twoD_pts = create_twoD_slab(l_bp_po1, l_p_po);
    l_po1_go = compute_orientation(l_bp_po1);

    # import plotting_routines as pr;
    # pr.plot_2d_pts_box(twoD_pts, l2D_bp_po1[0:2,:]);

    ################################################################################
    #### Replicate the 2D points in 3D (either in +Z or -Z)
    avec = l_bpb_p1[:,0]; bvec = l_bpb_p1[:,1];
    l_p2_p1 = find_int_solns(avec, bvec);
    l_p2_po1 = (l_p_po*l_p2_p1);
    l_p2_go = (l_po1_go*l_p2_po1);

    tz_vec = l_p2_go[:,2];
    threeD_pts, sim_cell = create_threeD_slab(zCut, tz_vec, l2D_bp_po1, twoD_pts);

    if cryst_typ == 'upper':
        tol1 = 1e-8; ind1 = np.where(threeD_pts[:,2] >= -tol1)[0];
        threeD_pts = threeD_pts[ind1,:];
    if cryst_typ == 'lower':
        tol1 = 1e-8; ind1 = np.where(threeD_pts[:,2] <= tol1)[0];
        threeD_pts = threeD_pts[ind1,:];

    return threeD_pts, sim_cell;

def get_gb_uID(l_bp_po1, l_p2_p1, l_p_po, bp_symm_grp, symm_grp_ax, sig_id):
    # hkl_p1 = compute_hkl_p(l_bp_po1, l_p_po);
    # hkl_po1 = l_p_po*hkl_p1;
    bpn_go1 = l_bp_po1[:,0].cross(l_bp_po1[:,1]);
    bpn_go1 = (np.array(bpn_go1/bpn_go1.norm(),dtype='double')).T;
    bp_fz_norms_go1, bp_fz_stereo = pfb.pick_fz_bpl(bpn_go1, bp_symm_grp, symm_grp_ax, 1e-04)
    bp_fz_go1 = iman.int_finder(bp_fz_norms_go1);

    bp_fz_go1 = bp_fz_go1.reshape(3,1);

    l_po_p = l_p_po.inv();
    l_po2_po1 = l_p_po*l_p2_p1*l_po_p
    l_po1_po2 = l_po2_po1.inv()

    bp_go2 = -l_po1_po2*bp_fz_go1;
    bp_go2 = iman.int_finder(bp_go2);


    gb_id = 'Al_S'+sig_id+'_N1_';
    gb_id = gb_id + str(bp_fz_go1[0]) + '_';
    gb_id = gb_id + str(bp_fz_go1[1]) + '_';
    gb_id = gb_id + str(bp_fz_go1[2]) + '_';
    gb_id = gb_id + 'N2_';
    gb_id = gb_id + str(bp_go2[0]) + '_';
    gb_id = gb_id + str(bp_go2[1]) + '_';
    gb_id = gb_id + str(bp_go2[2]);

    return gb_id;

def wrap_cc(cell1, pts):
    """
    Function finds the indices of atoms making tetrahedrons
    Parameters
    -------------
    cell1 :
        The simulation cell ( a 3*4 numpy array where the first 3 columns are the
        cell vectors and the last column is the box origin)
    pts :
        Position of atoms in initial cell, the atoms within an rCut of the initial cell.
    Returns
    ------------
    pts1 :
        Position of atoms in initial cell, the atoms within an rCut of the initial cell,
        and the Voronoi coordinates as the new set of atoms.
    """

    cell = ovd.SimulationCell(pbc = (True, True, False))
    cell[:,0] = cell1[:,0]
    cell[:,1] = cell1[:,1]
    cell[:,2] = cell1[:,2]
    cell[:,3] = cell1[:,3]

    data = ovd.DataCollection()
    data.objects.append(cell)
    particles = ovd.Particles()
    particles.create_property('Position', data=pts)
    data.objects.append(particles)

    # Create a new Pipeline with a StaticSource as data source:
    pipeline = Pipeline(source=StaticSource(data=data))

    pipeline.modifiers.append(ovm.WrapPeriodicImagesModifier())
    data1 = pipeline.compute()
    pts1 = np.array(data1.particle_properties['Position'][...])
    return pts1


def remove_periodic_overlaps(twoD_pts, twoDSig_mat):
    twoD_pts_sc1 = change_basis(twoD_pts, twoDSig_mat);
    twoD_pts_sc1 = cut_box_pts(twoD_pts_sc1, 1e-8);
    twoDSig_mat_inv = twoDSig_mat.inv();
    twoD_pts = change_basis(twoD_pts_sc1, twoDSig_mat_inv);

    if isinstance(twoDSig_mat, Matrix):
        twoD_mat = np.array(twoDSig_mat, dtype='double');


    # Use ovito nearest-neighbor alogrithm to remove overlaps

    cell = ovd.SimulationCell(pbc = (True, True, False))
    cell.is2D = True;
    cell[0:2,0] = twoD_mat[:,0];
    cell[0:2,1] = twoD_mat[:,1];

    npts = np.shape(twoD_pts)[0];
    pts1 = np.zeros((npts,3));
    pts1[:,0] = twoD_pts[:,0];
    pts1[:,1] = twoD_pts[:,1];

    data = ovd.DataCollection()
    data.objects.append(cell)
    particles = ovd.Particles()
    particles.create_property('Position', data=pts1)
    data.objects.append(particles)

    rCut = 0.01;

    finder = CutoffNeighborFinder(rCut, data);
    num_n = np.zeros((npts,1), dtype='int64');
    nn_inds = []; max_nn = 100;
    nn_inds_arr = np.ones((npts, max_nn), dtype='int64');
    nn_inds_arr = 2*int(npts)*nn_inds_arr;
    ct3 = 0;
    # Loop over all particles:
    for index in range(data.particles.count):
        # print("Neighbors of particle %i:" % index);
        num_neigh = 0; n_inds = [];
        n_inds.append(index);
        # Iterate over the neighbors of the current particle:
        for neigh in finder.find(index):
            num_neigh = num_neigh + 1;
            n_inds.append(neigh.index);
        num_n[index] = num_neigh;
        if num_neigh > 0:
            nn_inds.append(n_inds);
            nn_inds_arr[ct3,0:len(n_inds)] = np.array(n_inds);
            ct3 = ct3 + 1;

    nn_inds_arr = np.unique(np.sort(nn_inds_arr[:ct3,:max(num_n)[0]+1]), axis=0)
    inds1 = np.unique((nn_inds_arr[:,1:]).flatten());
    tinds1 = np.where(inds1 == 2*int(npts))[0];
    if np.size(tinds1) > 0:
        # inds1 = np.delete(inds1, -1);
        inds1 = np.delete(inds1, tinds1);
    twoD_pts = np.delete(twoD_pts, inds1, 0);

    return twoD_pts;

    # twoD_data = {};
    # twoD_data['pts'] = twoD_pts;
    # twoD_data['cell'] = twoD_mat;
    # import pickle as pkl;
    # pkl_name = 'twoD_data.pkl';
    # jar = open(pkl_name,'wb'); pkl.dump(twoD_data,jar);
    # jar.close();
    # import scipy.io as sio;
    # mat_name = 'twoD_data.mat';
    # sio.savemat(mat_name, twoD_data);
