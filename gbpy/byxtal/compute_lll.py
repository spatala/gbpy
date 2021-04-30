#!/usr/bin/env sage

############################################################
### SAGEmath routines for computing the reduced lattices ###
############################################################

import sys
from sage.all import *
import numpy as np
import sage_util_funcs as suf

#############################################
### Get input from stdin
args = len(sys.argv)

Sz = np.zeros((2,), dtype='int64')
Sz[0] = int(sys.argv[1])
Sz[1] = int(sys.argv[2])



nsz = Sz[0]*Sz[1]
A_lst = []
for ct1 in range(nsz):
   arg1 = int(sys.argv[ct1+3])
   A_lst.append(arg1)
#############################################

#############################################
### Convert input list to matrix
A = np.zeros((Sz[0], Sz[1]))

ct1 = 0
for i1 in range(Sz[0]):
   for j1 in range(Sz[1]):
      A[i1, j1] = A_lst[ct1]
      ct1 = ct1 + 1

A = Matrix(np.array(A, dtype='int64'))
#############################################

M3 = suf.compute_lll(A)

#############################################
### Print output to stdout
str1 = ''
for i1 in range(Sz[0]):
   for j1 in range(Sz[1]):
      str1 = str1 + ' ' + str(M3[i1, j1])

print(str1)
#############################################
