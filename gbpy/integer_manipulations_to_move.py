import gbpy.util_funcs as uf
import numpy as np

from sympy import Rational
from sympy.matrices import Matrix, eye, zeros;
from sympy import nsimplify
import sympy as spy

def check_int_mat(T, tol1):
    if isinstance(T, Matrix):
        T = np.array(T, dtype='double');
    return (np.max(np.abs(T - np.around(T))) < tol1);

def rat_approx(Tmat, tol1=0.01):
    """
    """
    input1 = Tmat.flatten()
    nshape = np.shape(Tmat)
    denum_max = 1/tol1
    Sz = input1.shape
    Nmat = np.zeros(np.shape(input1), dtype='int64')
    Dmat = np.zeros(np.shape(input1), dtype='int64')
    for ct1 in range(Sz[0]):
        num1 = (Rational(input1[ct1]).limit_denominator(denum_max))
        Nmat[ct1] = num1.p
        Dmat[ct1] = num1.q

    Nmat1 = np.reshape(Nmat, nshape)
    Dmat1 = np.reshape(Dmat, nshape)

    Nmat1 = np.array(Nmat1, dtype='int64')
    Dmat1 = np.array(Dmat1, dtype='int64')

    return Nmat1, Dmat1;

def gcd_arr(int_mat):
    input1 = int_mat.flatten()
    Sz = input1.shape
    gcd1 = 0
    for ct1 in range(Sz[0]):
        gcd1 = spy.gcd(gcd1, input1[ct1])

    return int(gcd1)

def lcm_arr(Dmat):
    input1 = Dmat.flatten()
    Sz = input1.shape
    lcm1 = 1
    for ct1 in range(Sz[0]):
        lcm1 = spy.lcm(lcm1, input1[ct1])

    return int(lcm1)

def int_approx(Tmat, tol1=0.01):
    tct1 = np.max(np.abs(Tmat))
    tct2 = np.min(np.abs(Tmat))

    mult1 = 1/((tct1 + tct2)/2)
    mult2 = 1/np.max(np.abs(Tmat))

    Tmat1 = Tmat*mult1
    Tmat2 = Tmat*mult2

    N1, D1 = rat_approx(Tmat1, tol1)
    N2, D2 = rat_approx(Tmat2, tol1)

    lcm1 = lcm_arr(D1)
    lcm2 = lcm_arr(D2)

    int_mat1 = np.array((N1/D1)*lcm1, dtype='double')
    int_mat2 = np.array((N2/D2)*lcm2, dtype='double')

    cond1 = check_int_mat(int_mat1, tol1*0.01)
    if cond1:
        int_mat1 = np.around(int_mat1)
        int_mat1 = np.array(int_mat1, dtype='int64')
    else:
        raise Exception("int_mat1 is not an integer matrix")

    cond2 = check_int_mat(int_mat2, tol1*0.01)
    if cond2:
        int_mat2 = np.around(int_mat2)
        int_mat2 = np.array(int_mat2, dtype='int64')
    else:
        raise Exception("int_mat2 is not an integer matrix")

    gcd1 = gcd_arr(int_mat1)
    gcd2 = gcd_arr(int_mat2)

    int_mat1 = int_mat1/gcd1
    int_mat2 = int_mat2/gcd2

    int_mat1 = np.array(int_mat1, dtype='int64')
    int_mat2 = np.array(int_mat2, dtype='int64')

    t1_mult = mult1*lcm1/gcd1
    t2_mult = mult2*lcm2/gcd2

    err1 = np.max(np.abs(Tmat - int_mat1/t1_mult))
    err2 = np.max(np.abs(Tmat - int_mat2/t2_mult))

    if err1 == err2:
        tnorm1 = np.linalg.norm(int_mat1)
        tnorm2 = np.linalg.norm(int_mat2)
        if (tnorm1 > tnorm2):
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult
    else:
        if err1 > err2:
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult
