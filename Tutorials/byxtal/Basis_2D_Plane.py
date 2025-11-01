import sys
# sys.path.insert(0, './../../../../gbpy/')
import numpy as np
# from sympy.matrices import Matrix, eye, zeros
import numpy.linalg as nla

import gbpy.byxtal.find_csl_dsc as fcd
import gbpy.byxtal.integer_manipulations as iman


# l_p_po = 1.0 * Matrix([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])
l_p_po = 1.0 * np.array([[0.,0.5,0.5],[0.5,0.,0.5],[0.5,0.5,0.]])
l_p_po


l_rp_po = fcd.reciprocal_mat(l_p_po)
l_rp_po

l_po_rp = nla.inv(l_rp_po)
n_po = np.array([[2], [3], [1]])
# n_rp = l_po_rp*n_po
n_rp = np.dot(l_po_rp,n_po)
print(n_rp)


ni_rp = iman.int_finder(n_rp)
ni_rp

import gbpy.byxtal.bp_basis as bpb
# l_2D_p = Matrix(bpb.bp_basis(ni_rp))
l_2D_p = bpb.bp_basis(ni_rp)
l_2D_p

l_2D_po = l_p_po*l_2D_p
l_2D_po