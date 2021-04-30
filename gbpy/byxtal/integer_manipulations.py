import numpy as np
from sympy import Rational
import sympy as spy
from fractions import Fraction
import numpy.linalg as nla


def gcd_vec(int_mat):
    """
    The function computes the GCD of an integer number.

    Parameters
    ----------
    int_mat: int
        Input integer number

    Returns
    -------
    gcd1: int
        The reatest common divisor of the input number.
        
    """
    input1 = int_mat.flatten()
    Sz = input1.shape
    gcd1 = 0
    for ct1 in range(Sz[0]):
        gcd1 = spy.gcd(gcd1, input1[ct1])

    return int(gcd1)


def gcd_array(input, order='all'):
    """
    The function computes the GCD of an array of numbers.

    Parameters
    ----------
    input : numpy.array or list
        Input n-D array of integers (most suitable for 1D and 2D arrays)
    order : {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Agcd: numpy.array
        An array of greatest common divisors of the input

    Notes
    -------
    * If order = **all**, the input array is flattened and the GCD is calculated
    * If order = **rows**, GCD of elements in each row is calculated
    * If order = **columns** or **cols**, GCD of elements in each column is calculated

    See Also
    --------
    gcd_vec: from fractions module for computing gcd of two integers

    """

    input = np.array(input)
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Agcd = gcd_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((sz1, 1))
        for ct1 in range(sz1):
            tmp_row = input[ct1, :]
            Agcd[ct1] = gcd_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Agcd = np.zeros((1, sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Agcd[0, ct1] = gcd_vec(tmp_row)
    return Agcd


def lcm_vec(Dmat):
    """
    The function computes the least common multiple (LCM).

    Parameters
    ----------
    Dmat: int
        The input number.

    Returns
    -------
    lcm1: int
        The least common multiple of the input number.
    """
    input1 = Dmat.flatten()
    Sz = input1.shape
    lcm1 = 1
    for ct1 in range(Sz[0]):
        lcm1 = spy.lcm(lcm1, input1[ct1])

    return int(lcm1)


def lcm_array(input, order='all'):
    """
    The function computes the LCM of an array of numbers.

    Parameters
    ----------
    input: numpy.array or list of intgers
        Input n-D array of integers (most suitable for 1D and 2D arrays)
    order: {'rows', 'columns', 'cols', 'all'}, optional

    Returns
    -------
    Alcm: numpy.array
        An array of least common multiples of the input

    Notes
    -------
    * If order = **all**, the input array is flattened and the GCD is calculated
    * If order = **rows**, GCD of elements in each row is calculated
    * If order = **columns** or **cols**, GCD of elements in each column is calculated

    See Also
    --------
    lcm_vec: from fractions module for computing gcd of two integers
    """

    input = np.array(input)
    # Only integer values are allowed
    # if input.dtype.name != 'int64':
    if not np.issubdtype(input.dtype, np.integer):
        raise Exception("Inputs must be real integers.")

    order_options = ('rows', 'columns', 'cols', 'all')
    try:
        Keys = (order_options.index(order))
    except:
        raise Exception(err_msg)

    if (Keys == 3):
        Alcm = lcm_vec(input)
    if (Keys == 0):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((sz1, 1))
        for ct1 in range(sz1):
            tmp_row = input[ct1, :]
            Alcm[ct1] = lcm_vec(tmp_row)
    if ((Keys == 1) or (Keys == 2)):
        sz1 = np.shape(input)[0]
        sz2 = np.shape(input)[1]
        Alcm = np.zeros((1, sz2))
        for ct1 in range(sz2):
            tmp_row = input[:, ct1]
            Alcm[0, ct1] = lcm_vec(tmp_row)
    return Alcm


def check_int_mat(T, tol1):
    """
    The function checks whether matrix is integer.

    Parameters
    ----------
    T: numpy.array
        Input matrix
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    Boolean
        True: If the matrix has integer elements.
        False: If the matrix does not have integer elements.
    """
    if isinstance(T, Matrix):
        T = np.array(T, dtype='double')
    return (np.max(np.abs(T - np.around(T))) < tol1)


def rat_approx(Tmat, tol1=0.01):
    """
    The function approximates the input with a rational number.

    Parameters
    ----------
    Tmat: numpy.array
        Input
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    Nmat1: int
        The nominator of the approximated rational number.
    Dmat1: int
        The nominator of the approximated rational number.
    """
    Tmat = np.array(Tmat)
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

    return Nmat1, Dmat1


def int_approx(Tmat, tol1=0.01):
    """
    The function 

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    int_mat1:
    t1_mult:
    """
    Tmat = np.array(Tmat)
    tct1 = np.max(np.abs(Tmat))
    tct2 = np.min(np.abs(Tmat))

    mult1 = 1/((tct1 + tct2)/2)
    mult2 = 1/np.max(np.abs(Tmat))

    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, mult1, tol1)
    int_mat2, t2_mult, err2 = mult_fac_err(Tmat, mult2, tol1)

    if err1 == err2:
        tnorm1 = nla.norm(int_mat1)
        tnorm2 = nla.norm(int_mat2)
        if (tnorm1 > tnorm2):
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult
    else:
        if err1 > err2:
            return int_mat2, t2_mult
        else:
            return int_mat1, t1_mult


def int_mult_approx(Tmat, tol1=0.01):
    """
    The function 

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix
    tol1: float
        Tolerance with default value 0.01
    Returns
    -------
    int_mat1:
    t1_mult:
    """
    Tmat = np.array(Tmat)
    int_mat1, t1_mult, err1 = mult_fac_err(Tmat, 1, tol1)
    return int_mat1, t1_mult


def mult_fac_err(Tmat, mult1, tol1):
    """
    The function 

    Parameters
    ----------
    Tmat: numpy.array
        Transformation matrix
    mult1:
    tol1: float
        Tolerance
    Returns
    -------
    int_mat1:
    t1_mult:
    err1:
    """
    Tmat1 = Tmat*mult1
    N1, D1 = rat_approx(Tmat1, tol1)

    lcm1 = lcm_array(D1)
    N1 = np.array(N1, dtype='double')
    D1 = np.array(D1, dtype='double')

    int_mat1 = np.array((N1/D1)*lcm1, dtype='double')

    cond1 = check_int_mat(int_mat1, tol1*0.01)
    if cond1:
        int_mat1 = np.around(int_mat1)
        int_mat1 = np.array(int_mat1, dtype='int64')
    else:
        raise Exception("int_mat1 is not an integer matrix")
    gcd1 = gcd_vec(int_mat1)
    int_mat1 = int_mat1/gcd1

    int_mat1 = np.array(int_mat1, dtype='int64')
    t1_mult = mult1*lcm1/gcd1
    err1 = np.max(np.abs(Tmat - int_mat1/t1_mult))
    return int_mat1, t1_mult, err1


def int_finder(input_v, tol=1e-6, order='all', tol1=1e-6):
    """
    The function computes the scaling factor required to multiply the
    given input array to obtain an integer array. The integer array is
    returned.

    Parameters
    ----------
    input1: numpy.array
        input array
    tol: float
        tolerance with Default = 1e-06
    order: str
        choices are 'rows', 'columns', 'col', 'all'.
        If order = 'all', the input array is flattened and then scaled. This is default value.
        If order = 'rows', elements in each row are scaled
        If order = 'columns' or 'cols'', elements in each column are scaled
    tol1: float
        tolerance with Default = 1e-06

    Returns
    -------
    output: numpy.array
        An array of integers obtained by scaling input
    """

    input1 = np.array(input_v)
    Sz = input1.shape
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    if int_check(input1, 15).all():
        input1 = np.around(input1)
        # Divide by LCM (rows, cols, all) <--- To Do
        tmult = gcd_array(input1.astype(dtype='int64'), order)
        if (order == 'all'):
            input1 = input1 / tmult
        elif (order == 'rows'):
            tmult = np.tile(tmult, (np.shape(input1[1])))
            input1 = input1 / tmult
        elif (order == 'col' or order == 'cols' or order == 'columns'):
            tmult = np.tile(tmult, (np.shape(input1[0])[0], 1))
            input1 = input1 / tmult
        output_v = input1
        if len(Sz) == 1:
            output_v = np.reshape(output_v, (np.size(output_v),))
        return output_v
    else:
        #   By default it flattens the array (if nargin < 3)
        if order.lower() == 'all':
            if len(Sz) != 1:
                input1.shape = (1, Sz[0]*Sz[1])
        else:
            Switch = 0
            err_msg = "Not a valid input. For the third argument please"+ \
                      " choose either \"rows\" or \"columns\" keys for this function."
            order_options = ('rows', 'columns', 'col')
            try:
                Keys = (order_options.index(order.lower()))
            except:
                raise Exception(err_msg)

            if (Keys == 1) or (Keys == 2):
                if input1.shape[0] != 1:
                    # Handling the case of asking a row vector
                    # with the 'column' key by mistake.
                    input1 = input1.T
                    Switch = 1
            # Handling the case of asking a column
            # vector with the 'row' key by mistake.
            if (Keys == 0) and (input1.shape[1] == 1):
                input1 = input1.T
                Switch = 1

        if (abs(input1) < tol).all():
            excep1 = 'All the input components cannot' \
                     + 'be smaller than tolerance.'
            raise Exception(excep1)

        tmp = np.array((abs(input1) > tol1))
        Vec = 2 * abs(input1[::]).max() * np.ones(
            (input1.shape[0], input1.shape[1]))
        Vec[tmp] = input1[tmp]
        MIN = abs(Vec).min(axis=1)
        # Transposing a row to a column
        MIN.shape = (len(MIN), 1)
        input1 = input1 / np.tile(MIN, (1, input1.shape[1]))
        N, D = rat(input1, tol)
        N[~tmp] = 0 # <---- added
        D[~tmp] = 1 # <---- added
        lcm_rows = lcm_array(D, 'rows')
        lcm_mat = np.tile(lcm_rows, (1, input1.shape[1]))
        Rounded = (N * lcm_mat) / D
        output_v = Rounded

        # --------------------------
        if order.lower() == 'all':
            if len(Sz) != 1:
                output_v.shape = (Sz[0], Sz[1])
        else:
            if (Keys) == 1 or (Keys) == 2:
                output_v = output_v.T
            if Keys == 0 and Switch == 1:
                output_v = output_v.T

        if len(Sz) == 1:
            output_v = np.reshape(output_v, (np.size(output_v), ))

        return output_v


def int_check(input, precis=6):
    """
    Checks whether the input variable (arrays) is an interger or not.
    A precision value is specified and the integer check is performed
    up to that decimal point.

    Parameters
    ----------
    input : numpy.array or list
        Input n-D array of floats.
    precis : int
        Default = 6.
        A value that specifies the precision to which the number is an
        integer. **precis = 6** implies a precision of :math:`10^{-6}`.

    Returns
    -------
    cond: Boolean
        'True' if the element is an integer to a certain precision,
        'False' otherwise
    """

    var = np.array(input)
    tval = 10 ** -precis
    t1 = abs(var)
    cond = (abs(t1 - np.around(t1)) < tval)
    return cond


def rat(input, tol=1e-06):
    """
    The function returns a rational (p/q) approximation of a given
    floating point array to a given precision

    Parameters
    ----------
    input : numpy.array or list
        input which is real numbers.
    tol : float
        Tolerance, Default = 1e-06

    Returns
    -------
    N: numpy.array
        N contain the numerators (p) and denominators (q) of the
        rational approximations.
    D: numpy.array
        D contain the numerators (p) and denominators (q) of the
        rational approximations.
    """
    input1 = np.array(input)
    if np.ndim(input1) == 1:
        input1 = np.reshape(input1, (1, input1.shape[0]))

    ## Why is this case necessary?
    if input1.ndim == 0:
        input1 = np.reshape(input1, (1, 1))

    Sz = input1.shape
    N = np.zeros((Sz[0], Sz[1]), dtype='int64')
    D = np.zeros((Sz[0], Sz[1]), dtype='int64')
    nDec = int(1/tol)
    for i in range(Sz[0]):
        for j in range(Sz[1]):
            N[i, j] = (Fraction.from_float(input1[i, j]).
                       limit_denominator(nDec).numerator)
            D[i, j] = (Fraction.from_float(input1[i, j]).
                       limit_denominator(nDec).denominator)
    return N, D
