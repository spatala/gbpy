from sympy import Matrix, gcdex

def mod(a, b):
    """
    Returns a % b, with a % 0 = a.

    Parameters
    ----------
    a : int
        The first argument in a % b.
    b : int
        The second argument in a % b.

    Returns
    -------
    int
        `a` modulus `b`.

    Examples
    ---------
    >>> mod(5, 2)
    1
    >>> mod(5, 0)
    5
    """
    if b == 0:
        return a
    return a % b

def smith_normal_form(A, compute_unimod = True):
    """
    Compute U,S,V such that U*A*V = S.

    This algorithm computes the Smith normal form of an integer matrix.
    If `compute_unimod` is True, it returns matrices (U, S, V) such
    that U*A*V = S, where U and V are unimodular and S is in Smith normal
    form. If `compute_unimod` is False, it returns S and does not compute U
    and V.

    Parameters
    ----------
    A : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        The matrix to factor.
    compute_unimod : bool
        Whether or not to compute and return unimodular matrices U and V.

    Returns
    -------
    U : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix, i.e. integer matrix with determinant +/- 1.

    S : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        A matrix in Smith normal form.

    V : :py:class:`~sympy.matrices.dense.MutableDenseMatrix`
        An unimodular matrix, i.e. integer matrix with determinant +/- 1.

    Examples
    ---------
    >>> from sympy import Matrix
    >>> A = Matrix([[1, 2],
    ...             [3, 4]])
    >>> U, S, V = smith_normal_form(A)
    >>> # Verify that U and V are both unimodular
    >>> U.det() in [1, -1] and V.det() in [1, -1]
    True
    >>> # Verify the factorization
    >>> U * A * V == S
    True
    >>> # Compute without U and V, verify that the result is the same
    >>> K = smith_normal_form(A, compute_unimod=False)
    >>> K == S
    True
    """

    # Get size and set up the unimodular matrices U and V
    m, n = A.shape
    min_m_n = min(m, n)
    S = A.copy()
    if compute_unimod:
        U, V = Matrix.eye(m), Matrix.eye(n)

    def row_col_all_zero(matrix, f):
        """
        Check that all entries to the right of and below `f` are zero.
        """
        for entry in matrix[f, f + 1:]:
            if entry != 0:
                return False
        for entry in matrix[f + 1:, f]:
            if entry != 0:
                return False
        return True

    # Main loop, iterate over all sub-matrices to reduce
    f = 0
    while f < min_m_n:

        # While there are non-zero elements to reduce in row/column f
        # and the diagonal element is not positive
        while not (row_col_all_zero(S, f) and S[f, f] >= 0):

            # Find index pair of minimum non-zero entry (in absolute value)
            # in the sub-matrix S[f:, f:].
            indices = ((i, j) for j in range(f, n) for i in range(f, m))
            key_val_pairs = ((index, abs(S[index])) for index in indices
                             if abs(S[index]) != 0)
            (i, j), min_val = min(key_val_pairs, key=lambda k: k[1])

            # Permute S to move the minimal element to the pivot location
            S[f:, j], S[f:, f] = S[f:, f], S[f:, j]
            S[i, f:], S[f, f:] = S[f, f:], S[i, f:]
            if compute_unimod:
                V[:, j], V[:, f] = V[:, f], V[:, j]
                U[i, :], U[f, :] = U[f, :], U[i, :]

            # If the freshly permuted pivot is negative, make it positive
            if S[f, f] < 0:
                S[f:, f] = -S[f:, f]
                if compute_unimod:
                    V[:, f] = -V[:, f]

            # Reduce row f so every entry is smaller than pivot
            for k in range(f + 1, n):
                if S[f, k] == 0:
                    continue

                # Subtract a times column f from column k
                a = S[f, k] // S[f, f]
                S[f:, k] = S[f:, k] - a * S[f:, f]
                if compute_unimod:
                    V[:, k] = V[:, k] - a * V[:, f]

            # Reduce column f so every entry is smaller than pivot
            for k in range(f + 1, m):
                if S[k, f] == 0:
                    continue

                # Subtract a times row f from row k
                a = S[k, f] // S[f, f]
                S[k, f:] = S[k, f:] - a * S[f, f:]
                if compute_unimod:
                    U[k, :] = U[k, :] - a * U[f, :]

        f += 1

    # Enforce divisibility criterion using the 'divisibility transformation'
    # matrices.
    for f in range(min_m_n):
        for k in range(f + 1, min_m_n):

            # Divisibility criterion is fulfilled
            if mod(S[k, k], S[f, f]) == 0:
                continue

            # S[f, f] does not divide S[k, k]
            r, s = S[f, f], S[k, k]
            a, b, c = gcdex(r, s)
            S[f, f], S[k, k] = c, (r * s) // c
            if compute_unimod:
                V[:, f], V[:, k] = (V[:, f] + V[:, k],
                                    -b * (s / c) * V[:,f] + a * (r / c) * V[:, k])
                U[f, :], U[k, :] = (a * U[f, :] + b * U[k, :],
                                    -(s / c) * U[f, :] + (r / c) * U[k, :])

    if compute_unimod:
        return U, S, V
    else:
        return S
    

# if __name__== '__main__':
#     M = Matrix([[2, 3, 5], [5, 1, 3], [4, 3, 2]])
#     U, S, V = smith_normal_form(M)
#     print("U: ")
#     print(U)
#     print("\nS: ")
#     print(S)
#     print("\nV: ")
#     print(V)

