import numpy as np
from typing import Callable

N = 4

DIM = 2 * N - 1  #Dimension of the Hiblert space

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
PLUS = (X + 1j * Y) / 2
MINUS = (X - 1j * Y) / 2
PLUS_Y = np.array([[1, 1j], [1j, -1]]) / 2
MINUS_Y = np.array([[1, -1j], [-1j, -1]]) / 2
I = np.identity(2)
CONTROLLED_Y = np.kron(I, PLUS_Y) + np.kron(Z, MINUS_Y)



def gate(op, n):
    if n > DIM:
        raise ValueError("Operator index must be smaller than " + str(DIM))
    mat = 1
    for i in range(1, DIM + 1):
        curr = np.identity(2)
        if i == n:
            curr = op
        mat = np.kron(mat, curr)
    return np.array(mat, dtype=np.complex128)


def list_to_op(ops: list):
    """Takes a list of 2X2 operators and returns their tensor product in order."""
    assert len(ops) == DIM, "Incorrect number of operators in the list."
    mat = 1
    for op in ops:
        np.kron(mat, op)
    return mat


def dict_to_op(ops_dict: dict):
    """Takes a dict from indices to 2X2 ops and returns a tensor product of the ops in the corresponding indices"""
    mat = 1
    for i in range(1, DIM + 1):
        if i in ops_dict.keys():
            mat = np.kron(mat, ops_dict[i])
        else:
            mat = np.kron(mat, np.identity(2, dtype=np.complex128))
    return mat


def swap(n, m):
    if n > DIM:
        raise ValueError("Operator index must be smaller than " + str(DIM))
    mat = (np.identity(2 ** DIM) + np.dot(gate(Z, n), gate(Z, m))) / 2
    mat += np.dot(gate(PLUS, n), gate(MINUS, m)) + np.dot(gate(MINUS, n), gate(PLUS, m))
    return mat


def matter(n):
    """Returns the lattice index of the n-th matter qubit."""
    assert n >= 1 and n <= DIM, "matter index not in range"
    return (n - 1) * 2 + 1


def field(n):
    """Returns the lattice index of the n-th field qubit."""
    assert n >= 1 and n <= DIM - 1, "matter index not in range"
    return n * 2


def is_unitary(op):
    return np.all(np.dot(op, op.conj().T) == np.identity(op.shape[0]))


def commutator(a, b):
    return np.dot(a, b) - np.dot(b, a)


def verify_SU2_algebra(x, y, z):
    flag = np.all(commutator(x, y) == 1j * z)
    flag = flag and np.all(commutator(y, z) == 1j * x)
    flag = flag and np.all(commutator(x, z) == -1j * y)
    return flag


def is_hermitian(op):
    return np.all(op.conj().T == op)


def get_spectrum(H, constraint: Callable = None):
    """
    Gets the valid spectrum of a Hamiltonian.
    :param H: The Hamiltonian matrix as a 2^(2N-1) X 2^(2N-1) array
    :param constraint: A function that gets (eigvals, eigvecs) and returns 2^(N-1) valid eigvals and eigvecs
    :return: eigvals, eigvecs that passed the constraint.
    """
    eigvals, eigvecs = np.linalg.eigh(H)
    if constraint is not None:
        return constraint(eigvals, eigvecs)
    return eigvals, eigvecs


def method0_hamiltonian(m, J, h):
    H = 0
    for n in range(1, N):
        H += (m / 2) * ((-1) ** n) * gate(Z, matter(n))
        H += (-J / 2) * (dict_to_op({matter(n): X, matter(n + 1): X}) + dict_to_op({matter(n): Y, matter(n + 1): Y}))
        op_dict = dict()
        for j in range(1, n + 1):
            op_dict[matter(j)] = Z
        H += h * ((-1) ** n) * dict_to_op(op_dict)
    H += (m / 2) * ((-1) ** N) * gate(Z, N)
    return H


def method1_hamiltonian(m, J, h):
    H = 0
    for n in range(1, N):
        H += (m / 2) * ((-1) ** n) * gate(Z, matter(n))
        H += h * gate(Z, field(n))
        op_dict1 = {field(n): X, matter(n): Y, matter(n + 1): X}
        op_dict2 = {field(n): X, matter(n): X, matter(n + 1): Y}
        if n > 1:
            op_dict1[field(n - 1)] = Z
            op_dict2[field(n - 1)] = Z
        H += (J / 2) * (dict_to_op(op_dict1) - dict_to_op(op_dict2))
    H += (m / 2) * ((-1) ** N) * gate(Z, N)
    return H


def gauss_op(n):
    assert 1 <= n <= N, "n is not in range [1," + str(N) + "]."
    if n == 1:
        op_dict = {field(1): Z, matter(1): -Z}
    elif n == N:
        op_dict = {field(N-1): Z, matter(N): -Z}
    else:
        op_dict = {field(n-1): Z, field(n): Z, matter(n): -Z}
    return dict_to_op(op_dict)


def check_invariance(vecs, ops: list):
    """
    Checks which vectors are invariant under given operators.
    :param vecs: An (n,m) array where the columns are vectors.
    :param ops: A list of nXn matrices.
    :return: A size (m,) boolean array denoting which vectors are invariant under all operators in ops.
    """
    applied_ops = [op.dot(vecs) for op in ops]
    is_amp_invar = np.array([np.isclose(vecs, applied_op) for applied_op in applied_ops])
    is_invar = np.all(is_amp_invar, axis=(0, 1))
    return is_invar


def apply_gauss_constraint(eigvals, eigvecs):
    """
    Returns the eigenvals and eigvecs such that the eigvecs are invariant under all N gauss law transformations.
    :param eigvals: A size (2^(2N-1)) array of eigenvals.
    :param eigvecs: A size (2^(2N-1)) X (2^(2N-1)) array whose columns are eigenvecs.
    :return: (eigvals, eigvecs) where eigvals a size 2^(N-1) array and eigvecs is a (2^(2N-1)) X 2^(N-1) array
    """
    gauss_ops = [gauss_op(n) for n in range(1, N + 1)]
    is_invar = check_invariance(eigvecs, gauss_ops)
    valid_eigvecs = np.take_along_axis(eigvecs, np.argwhere(is_invar).T, axis=1)
    return eigvals[is_invar], valid_eigvecs


def apply_ferm_parity_constraint(eigvals, eigvecs):
    """
    Returs eigvals and eigvecs such that the eigvecs are invariant under the global fermionic parity transformation.
    """
    op_dict = dict()
    for n in range(1, N+1):
        op_dict[matter(n)] = -Z
    parity_op = dict_to_op(op_dict)
    # field_ops = [gate(Z, field(i)) for i in range(1, N)]
    # is_invar = check_invariance(eigvecs, [parity_op] + field_ops)
    is_invar = check_invariance(eigvecs, [parity_op])
    valid_eigvecs = np.take_along_axis(eigvecs, np.argwhere(is_invar).T, axis=1)
    return eigvals[is_invar], valid_eigvecs



if __name__ == '__main__':
    H0 = method0_hamiltonian(1, 1, 1)
    H1 = method1_hamiltonian(1, 1, 1)
    eigvals0, eigvecs0 = get_spectrum(H0, apply_ferm_parity_constraint)
    eigvals1, eigvecs1 = get_spectrum(H1, apply_gauss_constraint)
    print(eigvals0)
    print(eigvals1)

