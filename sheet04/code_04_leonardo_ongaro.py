"""
Contents 
---------
- my_kron: tensor product 
- outer_product: outer product 
- embed_operator: embdes a local operator into the N-site space
- check_density_matrix: checks that a matrix is a valid density matrix 
- purity: Tr(rho^2)
- partial_trace: partial trace 
- partial_trace_pure: partial trace computed directly from the state vector 
- expect_full: expectation value of the full operator
- expect_reduced: expectation value computer with the reduced density matrix 
- compare_partial_trace_time: timing comparison between the two partial traces

The main compares the 3-qubit GHZ state with the statistical mixture 0.5*|000><000| + 0.5*|111><111| through expectation values, purity and reduced density matrices
"""

import numpy as np
import time 

# PAULI MATRICES

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# TENSOR PRODUCT

def my_kron(A, B):
    """
    Kronecker (tensor) product of two arrays, implemented block by block.
 
    1D vectors are treated as column vectors (n, 1), so the result is
    always a 2D matrix.
 
    Parameters
    ----------
    A, B : array_like with ndim 1 or 2, numeric values (complex allowed).
 
    Returns
    -------
    numpy.ndarray of shape (n_A*n_B, m_A*m_B).
 
    Raises
    ------
    TypeError / ValueError if the inputs are not numeric 1D or 2D arrays.
    """
    if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(B.dtype, np.number):
        raise TypeError(f"A and B must contain only numbers (found dtype: {A.dtype}).")
    if A.ndim not in [1,2] or B.ndim not in [1,2]:
        raise ValueError(f"A and B must have ndim 1 or 2, found {A.ndim} and {B.ndim}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError(f"A or B contain NaN or infinite values.")


    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n_A, m_A = A.shape 
    n_B, m_B = B.shape 

    # Result dtype is complex if at least one input is 
    C = np.zeros((n_A*n_B, m_A*m_B), dtype=np.result_type(A, B))

    for i in range(n_A):
        for j in range(m_A):
            C[i*n_B:(i+1)*n_B, j*m_B:(j+1)*m_B] = A[i, j] * B

    return C 

def outer_product(A, B):
    """
    Outer product C[i, j] = A[i] * B[j] computed with explicit loops.
 
    Parameters
    ----------
    A, B : array_like
        Vectors.
 
    Returns
    -------
    numpy.ndarray of shape (len(A), len(B)).
    Note: for complex states the bra is NOT conjugated; to obtain |psi><psi|
    call outer_product(psi, psi.conj()).
 
    Raises
    ------
    TypeError / ValueError if the inputs are not numeric vectors.
    """

    if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(B.dtype, np.number):
        raise TypeError(f"A and B must contain only numbers (found dtype: {A.dtype}).")
    if A.ndim != 1 or B.ndim != 1:
        raise ValueError(f"A and B must have ndim 1, found {A.ndim} and {B.ndim}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError(f"A or B contain NaN or infinite values.")

    C = np.zeros((len(A), len(B)), dtype=np.result_type(A, B))

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = A[i] * B[j]

    return C

def embed_operator(op, sites, N):
    """
    Embed a local operator into the space of N sites.
 
    `op` is placed on every site in `sites`, the identity on all the others:
    the result is the tensor product op_s1 (x) op_s2 (x) ... (x) I ...
 
    Parameters
    ----------
    op : array_like, square matrix (n, n)
        Local operator.
    sites : list of int
        Sites (0-indexed, distinct) on which op acts.
    N : int
        Total number of sites (N >= 1).
 
    Returns
    -------
    numpy.ndarray of shape (n**N, n**N).
 
    Raises
    ------
    TypeError / ValueError if op is not square, N is not a positive integer,
    or sites is invalid.
    """
    if op.shape[0] != op.shape[1]:
        raise ValueError(f"The operator must be square, found shape {op.shape}.")
    if not np.issubdtype(op.dtype, np.number):
        raise TypeError(f"The operator must contain only numbers (found dtype: {op.dtype}).")
    if op.ndim != 2:
        raise ValueError(f"The operator must have ndim 2, found {op.ndim}.")
    if not np.all(np.isfinite(op)):
        raise ValueError(f"A or B contain NaN or infinite values.")

    if (not isinstance(N, (int, np.integer)) and not isinstance(N, (bool, np.bool_))) or N < 1:
        raise ValueError("'N' must be an integer >= 1.")

    if not isinstance(sites, (list, tuple, np.ndarray)):
        raise TypeError(f"'The sites must be a list/tuple of integers.")
    sites = list(sites)
    if len(sites) == 0:
        raise ValueError(f"The sites list cannot be empty.")
    if not all(not isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in sites):
        raise TypeError(f"The sites list must contain only integers.")
    if any(s < 0 or s >= N for s in sites):
        raise ValueError(f"The sites contains indices outside the range [0, {N - 1}].")

    n = op.shape[0]
    I = np.eye(n)
    result = None  

    for site in range(1, N+1):
        local_op = op if site in sites else I
        result = local_op if result is None else my_kron(result, local_op)
    return result 

def check_density_matrix(A):
    """
    Check that A is a valid density matrix: unit trace, Hermitian and
    positive semidefinite (within the tolerance tol).
 
    Parameters
    ----------
    A : array_like, square matrix.
    tol : float >= 0, numerical tolerance.
 
    Returns
    -------
    bool
 
    Raises
    ------
    TypeError / ValueError if A is not a numeric square matrix or tol is invalid.
    """

    if np.trace(A) != 1:
        return False 
    if A != A.conjugate:
        return False 

    return True


def purity(A):
    A2 = np.matmul(A,A)
    return np.trace(A2) 

def partial_trace(rho, keep, dims):
    N = len(dims)
    D = int(np.prod(dims))

    rho = np.asarray(rho)

    trace = [i for i in range(N) if i not in keep]
    d_keep = int(np.prod([dims[k] for k in keep]))  
    d_trace = int(np.prod([dims[t] for t in trace]))

    rho = rho.reshape(dims + dims)
    perm = keep + [N + k for k in keep] + trace + [N + t for t in trace]
    rho = rho.transpose(perm).reshape(d_keep, d_keep, d_trace, d_trace)

    return np.trace(rho, axis1=2, axis2=3)

def expect_full(rho, op, sites, N):
    O_full = embed_operator(op, sites, N)          
    return np.trace(rho @ O_full)

def expect_reduced(rho, op, sites, dims):
    rho_A = partial_trace(rho, sites, dims)        
    return np.trace(rho_A @ op)

def partial_trace_pure(psi, keep, dims):
    N = len(dims)
    psi = np.asarray(psi).reshape(-1)
    trace = [i for i in range(N) if i not in keep]
    d_keep = int(np.prod([dims[k] for k in keep])) if keep else 1
    M = psi.reshape(dims).transpose(keep + trace).reshape(d_keep, -1)
    return M @ M.conj().T

def compare_partial_trace_time(psi, rho, keep, dims, repeats=5):
    """Time partial_trace(rho) vs partial_trace_pure(psi); return times, speedup, max difference."""
    dims, keep = list(dims), list(keep)
    D = int(np.prod(dims))
    psi = np.asarray(psi).reshape(-1)
    rho = np.asarray(rho)
    if psi.size != D or rho.shape != (D, D):
        raise ValueError("psi must have length prod(dims), rho shape (prod(dims),)*2")

    def best_time(f):
        best = np.inf
        for _ in range(repeats):
            t = time.perf_counter()
            out = f()
            best = min(best, time.perf_counter() - t)
        return out, best

    r_rho, t_rho = best_time(lambda: partial_trace(rho, keep, dims))
    r_psi, t_psi = best_time(lambda: partial_trace_pure(psi, keep, dims))
    diff = float(np.max(np.abs(r_rho - r_psi)))

    print(f"N={len(dims)}, keep={keep}: rho route {t_rho:.3e} s | psi route {t_psi:.3e} s "
          f"| speedup x{t_rho/t_psi:.1f} | max diff {diff:.1e}")
    return {"t_rho": t_rho, "t_psi": t_psi, "speedup": t_rho / t_psi, "max_abs_diff": diff}



def main():
    #------------------------
    # TENSOR PRODUCT
    #------------------------

    A1 = np.random.rand(3,3)
    B1 = np.random.rand(3,3)

    assert(np.allclose(my_kron(A1, B1), np.kron(A1, B1)))

    A2 = np.random.rand(4,4)
    B2 = np.random.rand(4,4)

    assert(np.allclose(my_kron(A2, B2), np.kron(A2, B2)))

    print(f"The tensor product of {A1} and {A2} is {my_kron(A1, B1)}")

    #-----------------------
    # MANY BODY STATES 
    #-----------------------

    v0 = np.array([1, 0])
    v1 = np.array([0,1])
    v000 = my_kron(v0, my_kron(v0, v0)) 
    v111 = my_kron(v1, my_kron(v1, v1)) 

    ghz = 1/np.sqrt(2) (v000 + v111)
    rho_ghz = outer_product(ghz, ghz)
    assert(np.allclose(rho_ghz, np.outer(ghz, ghz)))
    rho_mix = 0.5 * outer_product(v000, v000) + 0.5 * outer_product(v111, v111)
    assert(np.allclose(rho_mix, 0.5 * np.outer(v000, v000) + 0.5 * np.outer(v111, v111)))

    print(f"The density matrix of the ghz state is {rho_ghz}")
    print(f"The density matrix of the mixed state is {rho_mix}")

    print(f"Is rho_ghz a valid density matrix? {check_density_matrix(rho_ghz)}")
    print(f"Is tho_mix a valid density matrix? {check_density_matrix(rho_mix)}")

    print(f"The purity of rho_ghz is {purity(rho_ghz)}")
    print(f"The purity of rho_mix is {purity(rho_mix)}")

    #----------------------
    # GHZ vs. the mixed state 
    #----------------------

    i = 1 
    j = 2 
    k = 3

    sigma_z_i = embed_operator(sigma_z, [i], 3)
    sigma_z_j = embed_operator(sigma_z, [j], 3)
    sigma_x_i = embed_operator(sigma_x, [i], 3)
    sigma_x_j = embed_operator(sigma_x, [j], 3)
    sigma_x_k = embed_operator(sigma_x, [k], 3)

    sigma_z_i_ghz = np.trace(sigma_z_i @ rho_ghz)
    sigma_z_i_mix = np.trace(sigma_z_i @ rho_mix)
    print(f"The mean value of sigma_z^i in rho_ghz is {sigma_z_i_ghz}")
    print(f"The mean value of sigma_z^i in rho_mix is {sigma_z_i_mix}")

    sigma_x_j_ghz = np.trace(sigma_x_j @ rho_ghz)
    sigma_x_j_mix = np.trace(sigma_x_j @ rho_mix)
    print(f"The mean value of sigma_x^j in rho_ghz is {sigma_x_j_ghz}")
    print(f"The mean value of sigma_x^j in rho_mix is {sigma_x_j_mix}")

    sigma_z_ij_ghz = np.trace(sigma_z_i @ sigma_z_j @ rho_ghz)
    sigma_z_ij_mix = np.trace(sigma_z_i @ sigma_z_j @ rho_mix)
    print(f"The mean value of sigma_z^i * sigma_z^j in rho_ghz is {sigma_z_ij_ghz}")
    print(f"The mean value of sigma_z^i * sigma_z^j in rho_mix is {sigma_z_ij_mix}")

    sigma_x_ijk_ghz = np.trace(sigma_x_i @ sigma_x_j @ sigma_x_k @ rho_ghz)
    sigma_x_ijk_mix = np.trace(sigma_x_i @ sigma_x_j @ sigma_x_k @ rho_mix)
    print(f"The mean value of sigma_x^i * sigma_x^j * sigma_x^k in rho_ghz is {sigma_x_ijk_ghz}")
    print(f"The mean value of sigma_x^i * sigma_x^j * sigma_x^k in rho_mix is {sigma_x_ijk_mix}")

    #--------------------
    # PARTIAL TRACE 
    #--------------------

    i = 1 
    j = 2 

    print(f"The single site reduced matrix of rho_ghz at the site {i} is {partial_trace(rho_ghz, [i], 3)}")
    print(f"The single site reduced matrix of rho_mix at the site {i} is {partial_trace(rho_mix, [i], 3)}")

    print(f"The two-site reduced matrix of rho_ghz at the site {i} and {j} is {partial_trace(rho_ghz, [i, j], 3)}")
    print(f"The two-site reduced matrix of rho_mix at the site {i} and {j} is {partial_trace(rho_mix, [i, j], 3)}")

    #-------------------------------------------
    # TWO ROUTES TO THE SAME EXPECTATION VALUE 
    #-------------------------------------------

    print(f"Are the two methods equal, with single site and rho_ghz? {np.isclose(expect_full(rho_ghz, sigma_z, [i], 3), expect_reduced(rho_ghz, sigma_z, [i], 3))}")
    print(f"Are the two methods equal, with single site and rho_mix? {np.isclose(expect_full(rho_mix, sigma_z, [i], 3), expect_reduced(rho_mix, sigma_z, [i], 3))}")
    print(f"Are the two methods equal, with two sites and rho_ghz? {np.isclose(expect_full(rho_ghz, sigma_z, [i, j], 3), expect_reduced(rho_ghz, sigma_z, [i, j], 3))}")
    print(f"Are the two methods equal, with two sites and rho_mix? {np.isclose(expect_full(rho_mix, sigma_z, [i, j], 3), expect_reduced(rho_mix, sigma_z, [i, j], 3))}")


    #-------------------------------------
    # PARTIAL TRACE OF A PURE STATE 
    #-------------------------------------

    compare_partial_trace_time(ghz, rho_ghz, keep=[1, 2], dims=[2]*12)


