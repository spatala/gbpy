import sys
# @Leila: Why are you inserting this path?
# sys.path.insert(0, '../../../')
import byxtal.lattice as gbl
import byxtal.integer_manipulations as iman
import byxtal.bp_basis as bpb
import numpy as np
import math as mt
import gbpy.util_funcs_create_byxtal as uf
import gbpy.generate_hkl_indices as ghi
from sympy.matrices import Matrix, eye, zeros

## Sympy + SAGE routines
import integer_manipulations as iman
import util_funcs as uf


# get the lattice parameter of Al
l1 = gbl.Lattice('hp_id')
# Determine the primitive basis of fcc crystal structure
l_p_po = Matrix(l1.l_p_po)
l_po_p = l_p_po.inv()
## (h k l) indices of the fcc lattice (described in p)
h = 3
k = 4
l = 5

hkl_p = Matrix([h,k,l])


# hkl_p1
## l_bpb_p: Primitive Basis vectors of the boundary-plane (in p reference frame)
l_bpb_p = bpb.bp_basis(hkl_p)
l_bpb_p = Matrix(l_bpb_p.astype(int))
## l_bpb_p: Primitive Basis vectors of the boundary-plane (in po reference frame)
l_bpb_po = l_p_po*l_bpb_p

## Cut-off for area of the simulation box
lat_par = l1.lat_params['a']
r_cut = lat_par*4
A_cut = r_cut**2
## area_bpl: Area of the 2D-primitive-unit-cell
area_bpl = (l_bpb_po[:,0].cross(l_bpb_po[:,1])).norm()
# sig_num = np.ceil(A_cut/area_bpl)
sig_num = (A_cut/area_bpl).ceiling()


ind2 = np.array([], dtype='int64')
tol = 0.1
while (np.size(ind2) == 0):
    # Generate 2D hermite normal forms for sig_num (hnf_mats)
    hnf_mats = ghi.sig_hnf_mats(sig_num)
    # Compute the properties of the sub-lattices
    l_sig_p_mats, l_sig_po_mats = ghi.compute_hnf_props(hnf_mats, l_bpb_p, l_p_po, tol)
    # Get the index for the sub-lattice that has the minimum cost
    ind2 = ghi.ind_min_cost(l_sig_po_mats, r_cut)
    sig_num = sig_num + 1
