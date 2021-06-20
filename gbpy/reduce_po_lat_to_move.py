import subprocess
import numpy as np
import gbpy.integer_manipulations as iman
from sympy.matrices import Matrix, eye, zeros;

def call_sage_math(exec_str, inp_args):
    run_lst = []
    run_lst.append(exec_str)

    if len(inp_args.keys()) == 2:
        A = inp_args['mat']
        sig_num = inp_args['sig_num']
        str1 = str(sig_num)
        run_lst.append(str1)

    if len(inp_args.keys()) == 1:
        A = inp_args['mat']

    Sz = np.shape(A)
    run_lst.append(str(Sz[0]))
    run_lst.append(str(Sz[1]))

    for i1 in range(Sz[0]):
        for j1 in range(Sz[1]):
            str1 = str(A[i1, j1])
            run_lst.append(str1)

    result = subprocess.run(run_lst, stdout=subprocess.PIPE)
    str_out = (result.stdout).split()

    sz1 = len(str_out)
    M_out = np.zeros((Sz[0],Sz[1]), dtype='int64')
    ct1 = 0
    for i1 in range(Sz[0]):
        for j1 in range(Sz[1]):
            M_out[i1, j1] = int(str_out[ct1])
            ct1 = ct1 + 1

    return M_out

def reduce_po_lat(l_csl_p, l_p_po, tol):
    l_p_po = np.array(l_p_po, dtype='double')
    l_csl_po = l_p_po.dot(l_csl_p)
    lInt_csl_po, m1 = iman.int_approx(l_csl_po, tol)
    inp_args = {}
    inp_args['mat'] = lInt_csl_po
    lllInt_csl_po = call_sage_math('./compute_LLL.py', inp_args)
    lllInt_csl_po = Matrix(lllInt_csl_po)
    Sz = lllInt_csl_po.shape
    if Sz[0] == Sz[1]:
        if lllInt_csl_po.det() < 0:
            if Sz[0] == 3:
                M4 = Matrix([[0,1,0],[1,0,0],[0,0,1]])
                lllInt_csl_po = lllInt_csl_po*M4
            if Sz[0] == 2:
                M4 = Matrix([[0,1],[1,0]])
                lllInt_csl_po = lllInt_csl_po*M4

        Tmat = ((Matrix(lInt_csl_po)).inv())*(lllInt_csl_po)
    else:
        A1 = (np.array(lllInt_csl_po, dtype='int64'))
        A2 = (np.array(lInt_csl_po, dtype='int64'))
        # A1inv = np.linalg.pinv(A1)
        A2inv = np.linalg.pinv(A2)
        Tmat = Matrix(np.dot(A2inv, A1))

    cond1 = iman.check_int_mat(Tmat, 1e-12)
    if cond1:
        Tmat1 = Matrix(np.array(np.around(np.array(Tmat, dtype='double')), dtype='int64'))
        return Tmat1
    else:
        raise Exception("Tmat is not an integer matrix.")
