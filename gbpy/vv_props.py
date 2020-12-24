import numpy as np
from scipy.spatial.distance import pdist
from pyhull import qdelaunay
import ovito.data as ovd
from ovito.pipeline import StaticSource, Pipeline
import ovito.modifiers as ovm
import byxtal.lattice as gbl

def Circum_O_R(vertex_pos, tol):
    """
    Function finds the center and the radius of the circumsphere of the every tetrahedron.
    Reference:
    Fiedler, Miroslav. Matrices and graphs in geometry. No. 139. Cambridge University Press, 2011.

    Parameters
    -----------------
    vertex_pos : numpy array
        The position of vertices of a tetrahedron
    tol : float
        Tolerance defined  to identify co-planar tetrahedrons

    Returns
    ----------
    circum_center : numpy array
        The center of the circum-sphere
    circum_rad : float
        The radius of the circum-sphere
    """
    dis_ij = pdist(vertex_pos, 'euclidean')
    sq_12, sq_13, sq_14, sq_23, sq_24, sq_34 = np.power(dis_ij, 2)

    MatrixC = np.array([[0, 1, 1, 1, 1], [1, 0, sq_12, sq_13, sq_14], [1, sq_12, 0, sq_23, sq_24],
                        [1, sq_13, sq_23, 0, sq_34], [1, sq_14, sq_24, sq_34, 0]])

    det_MC = (np.linalg.det(MatrixC))

    if (det_MC < tol):
        return [0, 0, 0], 0
    else:
        M = -2*np.linalg.inv(MatrixC)
        circum_center = (M[0, 1]*vertex_pos[0, :] + M[0, 2]*vertex_pos[1, :] + M[0, 3]*vertex_pos[2, :] +
                         M[0, 4] * vertex_pos[3, :]) / (M[0, 1] + M[0, 2] + M[0, 3] + M[0, 4])
        circum_rad = np.sqrt(M[0, 0])/2

    return circum_center, circum_rad


def triang_inds(pts_w_imgs, gb1_inds, inds_arr):
    """
    Function finds the indices of atoms which make tetrahedrons with at least one GB atom.

    Parameters
    -------------
    pts_w_imgs : numpy array
        The position of atoms which are inside the main box and within rCut to the main box.
    gb1_inds : numpy array
        Indices of the GB atoms
    inds_arr : numpy array
        The atom indices of the initial unit cell with no replicates.

    Returns
    ------------
    tri_vertices : numpy array
        Tetrahedrons with at least one corner in the GB region.
    gb_tri_inds1 : numpy array
        The indices of atoms which make tri_vertices tetrahedrons.
    """
    tri_simplices = qdelaunay("i Qt", pts_w_imgs)
    num_tri = int(tri_simplices[0])
    tri_vertices = np.zeros((num_tri, 4), dtype='int')
    for ct1 in range(num_tri):
        tri_vertices[ct1, :] = np.array([int(i) for i in str.split(tri_simplices[ct1+1])])

    inds1 = (inds_arr[tri_vertices])
    inds2 = np.copy(inds1)
    inds2.sort(axis=1)
    tarr1 = np.zeros((np.shape(pts_w_imgs)[0], ))
    tarr1[gb1_inds] = 1
    gb_tri_inds = np.where(np.sum(tarr1[tri_vertices], axis=1))[0]
    i1, ia = np.unique(inds2[gb_tri_inds, :], return_index=True, axis=0)
    gb_tri_inds1 = gb_tri_inds[ia]

    return tri_vertices, gb_tri_inds1


def vv_props(pts_w_imgs, tri_vertices, gb_tri, l1):
    """
    Function finds the circum-center/sphere of tetrahedrons containing
    at least one GB atom.

    Parameters
    -------------
    pts_w_imgs : numpy array
        The position of atoms which are inside the main box and within rCut to the main box.
    tri_vertices : numpy array
        Tetrahedrons with at least one corner in the GB region.
    gb_tri : numpy array
        The indices of atoms which make tri_vertices tetrahedrons.
    lat_par : float
        Lattice parameter for the crystal being simulated.

    Returns
    ------------
    cc_coors1 : numpy array
        The coordinates of the circum-center of the tetrahedrons
    cc_rad1 : numpy array
        The circum-radius of the tetrahedrons.
    """
    num_tri = np.shape(gb_tri)[0]
    cc_coors = np.zeros((num_tri, 3))
    cc_rad = np.zeros((num_tri, 1))
    tol = 1e-10*(l1.lat_params['a']**3)
    ct1 = 0
    for tri1 in gb_tri:
        simplex = tri_vertices[tri1, :]
        vertex_pos = pts_w_imgs[simplex, :]
        [cc, cr] = Circum_O_R(vertex_pos, tol)
        cc_coors[ct1, :] = cc
        cc_rad[ct1, :] = cr
        ct1 = ct1 + 1
    tinds1 = np.where(cc_rad[:, 0] < 1e-12)[0]
    cc_coors1 = np.delete(cc_coors, tinds1, 0)
    cc_rad1 = np.delete(cc_rad, tinds1, 0)

    return cc_coors1, cc_rad1


def wrap_cc(cell1, pts):
    """
    Function finds the indices of atoms making tetrahedrons

    Parameters
    -------------
    cell1 : numpy array
        The simulation cell ( a 3*4 numpy array where the first 3 columns are the
        cell vectors and the last column is the box origin)
    pts : numpy array
        Position of atoms in initial cell, the atoms within an rCut of the initial cell.

    Returns
    ------------
    pts1 : numpy array
        Position of atoms in initial cell, the atoms within an rCut of the initial cell,
        and the Voronoi coordinates as the new set of atoms.

    """
    data = ovd.DataCollection()
    data.objects.append(cell1)
    particles = ovd.Particles()
    particles.create_property('Position', data=pts)
    data.objects.append(particles)

    # Create a new Pipeline with a StaticSource as data source:
    pipeline = Pipeline(source=StaticSource(data=data))

    pipeline.modifiers.append(ovm.WrapPeriodicImagesModifier())
    data1 = pipeline.compute()
    pts1 = np.array(data1.particle_properties['Position'][...])
    return pts1
