#!/bin/bash

import gbpy.byxtal.lll_tools as lt
import numpy as np

imat2 = np.array([[  9,  10], [  1,   0], [128, 160]])
lll_imat2 = np.array([[ 5,  1], [ 5, -1], [ 0, 32]])
l_dsc_po = np.array([[ 4.5,  5.0],[ 0.5,    0],[64.0, 80.0]])
lll_dsc_po = np.array([[ 2.5,  0.5], [ 2.5, -0.5], [ 0. , 16. ]])

cond1 = lt.check_basis_equi(imat2, lll_imat2)
cond2 = lt.check_basis_equi(l_dsc_po, lll_dsc_po)
