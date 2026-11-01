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
    Kronecker product of two arrays
 
    1D vectors are treated as column vectors (n, 1), so the result is
    always a 2D matrix.
 
    Parameters
    ----------
    A, B : array with ndim 1 or 2
 
    Returns
    -------
    numpy.ndarray of shape (n_A*n_B, m_A*m_B).

    """

    # INPUT CHECK
    if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(B.dtype, np.number):
        raise TypeError(f"A and B must contain only numbers (found dtype: {A.dtype}).")
    if A.ndim not in [1,2] or B.ndim not in [1,2]:
        raise ValueError(f"A and B must have ndim 1 or 2, found {A.ndim} and {B.ndim}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError(f"A or B contain NaN or infinite values.")


    if A.ndim == 1:
        A = A.reshape(-1, 1) # reshape the vector as a matrix
    if B.ndim == 1:
        B = B.reshape(-1, 1) # reshape the vector as a matrix

    # dimensions of A and B
    n_A, m_A = A.shape 
    n_B, m_B = B.shape 

    # Result dtype is complex if at least one input is 
    C = np.zeros((n_A*n_B, m_A*m_B), dtype=np.result_type(A, B))
    for i in range(n_A):
        for j in range(m_A):
            C[i*n_B:(i+1)*n_B, j*m_B:(j+1)*m_B] = A[i, j] * B           # block by block multiplication

    return C 

def outer_product(A, B):
    """
    Outer product C[i, j] = A[i] * B[j] computed with explicit loops.
 
    Parameters
    ----------
    A, B : vectors.
 
    Returns
    -------
    numpy.ndarray of shape (len(A), len(B))

    """

    # INPUT CHECK
    if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(B.dtype, np.number):
        raise TypeError(f"A and B must contain only numbers (found dtype: {A.dtype}).")
    if A.ndim != 1 or B.ndim != 1:
        raise ValueError(f"A and B must have ndim 1, found {A.ndim} and {B.ndim}.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
        raise ValueError(f"A or B contain NaN or infinite values.")

    B = B.conj()

    # Result dtype is complex if at least one input is 
    C = np.zeros((len(A), len(B)), dtype=np.result_type(A, B))

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = A[i] * B[j]

    return C

def embed_operator(op, sites, N):
    """
    Embed a local operator into the space of N sites.
 
    'op' is placed on every site in 'sites', the identity on all the others
 
    Parameters
    ----------
    op : square matrix (n, n)
        Local operator.
    sites : list of int
        Sites on which op acts.
    N : int
        Total number of sites (N >= 1).
 
    Returns
    -------
    numpy.ndarray of shape (n**N, n**N).

    """

    # INPUT CHECK
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

    if not isinstance(sites, list):
        raise TypeError(f"'The sites must be a list of integers.")
    if len(sites) == 0:
        raise ValueError(f"The sites list cannot be empty.")
    if not all(isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in sites):
        raise TypeError(f"The sites list must contain only integers.")
    if any(s < 0 or s >= N for s in sites):
        raise ValueError(f"The sites contains indices outside the range [0, {N - 1}].")

    n = op.shape[0]
    I = np.eye(n) # identity matrix of dimension n
    result = None  

    for site in range(0, N):
        local_op = op if site in sites else I
        result = local_op if result is None else my_kron(result, local_op)

    return result 

def check_density_matrix(A):
    """
    Check that A is a valid density matrix.
 
    Parameters
    ----------
    A : square matrix.
 
    Returns
    -------
    bool
 
    """

    # INPUT CHECK
    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("'A' must contain only numbers.")

    if A.ndim != 2:
        raise ValueError("'A' must be a 2D matrix.")

    if not np.all(np.isfinite(A)):
        raise ValueError("'A' contains NaN or infinite values.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("'A' must be square.")

    if not np.isclose(np.trace(A), 1):
        return False 
    if not np.allclose(A, A.conj().T):
        return False 

    return True


def purity(A):
    """
    Purity Tr(rho^2) of a density matrix
 
    Parameters
    ----------
    A : square matrix.
 
    Returns
    -------
    float
 
    """

    # INPUT CHECK
    if not np.issubdtype(A.dtype, np.number):
        raise TypeError("'A' must contain only numbers.")

    if A.ndim != 2:
        raise ValueError("'A' must be a 2D matrix.")

    if not np.all(np.isfinite(A)):
        raise ValueError("'A' contains NaN or infinite values.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("'A' must be square.")

    return np.trace(A @ A)


def partial_trace(rho, keep, dims):
    """
    Partial trace of a density matrix on a composite system.
 
    Parameters
    ----------
    rho : matrix (D, D) with D = product of dims.
    keep : list of int
    dims : list of int
        Dimension of each subsystem (e.g. [2, 2, 2] for 3 qubits).
 
    Returns
    -------
    numpy.ndarray of shape (d_keep, d_keep).
 
    """

    # INPUT CHECK
    if not isinstance(dims, list):
        raise TypeError("'dims' must be a list.")

    if len(dims) == 0:
        raise ValueError("'dims' cannot be empty.")

    if not all(isinstance(d, (int, np.integer)) and not isinstance(d, (bool, np.bool_)) and d >= 1 for d in dims):
        raise ValueError("'dims' must contain integers >= 1.")

    if not isinstance(keep, list):
        raise TypeError("'keep' must be a list.")

    if not all(isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in keep):
        raise TypeError("'keep' must contain only integers.")

    N = len(dims)

    if any(s < 0 or s >= N for s in keep):
        raise ValueError("'keep' contains invalid indices.")

    if not np.issubdtype(rho.dtype, np.number):
        raise TypeError("'rho' must contain only numbers.")

    if rho.ndim != 2:
        raise ValueError("'rho' must be a 2D matrix.")

    if not np.all(np.isfinite(rho)):
        raise ValueError("'rho' contains NaN or infinite values.")

    if rho.shape[0] != rho.shape[1]:
        raise ValueError("'rho' must be square.")

    D = int(np.prod(dims)) # dimension of the total density matrix

    if rho.shape != (D, D):
        raise ValueError(f"'rho' must have shape ({D}, {D}).")

    traced = [s for s in range(N) if s not in keep]                                 # indices of the subsystems to trace
    d_keep = int(np.prod([dims[s] for s in keep]))                              # dimension of the space kept 
    d_traced = int(np.prod([dims[t] for t in traced]))                      # dimension of the space traces 
    result = np.zeros((d_keep, d_keep), dtype=complex)

    # sum over all the states  in the traced subystems
    for j in range(d_traced):
        indices = np.unravel_index(j, [dims[t] for t in traced])                        # if the number of subsystems to trace is > 1 we need to consider more indices
        basis = dict(zip(traced, indices))                                          # dictionary with a link between indices and j

        K = np.eye(1)
        for s, d in enumerate(dims):
            if s in keep:
                K = my_kron(K, np.eye(d))
            else:
                K = my_kron(K, np.eye(d)[:, [basis[s]]])  

        result += K.conj().T @ rho @ K   # (I ⊗ <i|) rho (I ⊗ |i>)

    return result

def expect_full(rho, op, sites, N):
    """
    Expectation value Tr(rho * O), where O is 'op' embedded on 'sites' in the
    full space of N sites.
 
    Parameters
    ----------
    rho : density matrix (n**N, n**N).
    op : local operator (n, n), repeated on each site in 'sites'.
    sites : list of int.
    N : int, number of sites.
 
    Returns
    -------
    expectation value.
 
    """

    # INPUT CHECK
    O_full = embed_operator(op, sites, N)  

    if not np.issubdtype(rho.dtype, np.number):
        raise TypeError("'rho' must contain only numbers.")

    if rho.ndim != 2:
        raise ValueError("'rho' must be a 2D matrix.")

    if not np.all(np.isfinite(rho)):
        raise ValueError("'rho' contains NaN or infinite values.")

    if rho.shape != O_full.shape:
        raise ValueError(f"'rho' must have shape {O_full.shape}.")

    return np.trace(rho @ O_full)

def expect_reduced(rho, op, sites, dims):
    """
    Expectation value Tr(rho_A * op) using the reduced density matrix on 'sites'.

 
    Parameters
    ----------
    rho : full density matrix.
    op : matrix (d_keep, d_keep).
    sites : list of int of the subsystems on which op acts.
    dims : list of int, dimensions of the subsystems.
 
    Returns
    -------
    expectation value.
 
    """

    # INPUT CHECK
    if not isinstance(dims, list):
        raise TypeError("'dims' must be a list.")

    if len(dims) == 0:
        raise ValueError("'dims' cannot be empty.")

    if not all(isinstance(d, (int, np.integer)) and not isinstance(d, (bool, np.bool_)) and d >= 1 for d in dims):
        raise ValueError("'dims' must contain integers >= 1.")

    if not isinstance(sites, list):
        raise TypeError("'sites' must be a list.")

    sites = list(sites)

    if len(sites) == 0:
        raise ValueError("'sites' cannot be empty.")

    if not all(isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in sites ):
        raise TypeError("'sites' must contain only integers.")

    if any(s < 0 or s >= len(dims) for s in sites):
        raise ValueError("'sites' contains invalid indices.")

    if len(set(sites)) != len(sites):
        raise ValueError("'sites' contains repeated indices.")

    if not np.issubdtype(op.dtype, np.number):
        raise TypeError("'op' must contain only numbers.")

    if op.ndim != 2:
        raise ValueError("'op' must be a 2D matrix.")

    if not np.all(np.isfinite(op)):
        raise ValueError("'op' contains NaN or infinite values.")

    if op.shape[0] != op.shape[1]:
        raise ValueError("'op' must be square.")

    d_keep = int(np.prod([dims[s] for s in sites]))

    if op.shape != (d_keep, d_keep):
        raise ValueError(f"'op' must have shape ({d_keep}, {d_keep}).")
    
    rho_A = partial_trace(rho, sites, dims)        
    return np.trace(rho_A @ op)

def partial_trace_pure(psi, keep, dims):
    """
    Reduced density matrix of a pure state |psi>, without building |psi><psi|.
 
    Parameters
    ----------
    psi : vector of length D = product of dims.
    keep : list of int, subsystems to keep.
    dims : list of int, dimensions of the subsystems.
 
    Returns
    -------
    numpy.ndarray of shape (d_keep, d_keep).
 
    """

    if not isinstance(dims, list):
        raise TypeError("'dims' must be a list")


    if len(dims) == 0:
        raise ValueError("'dims' cannot be empty.")

    if not all(isinstance(d, (int, np.integer)) and not isinstance(d, (bool, np.bool_)) and d >= 1 for d in dims):
        raise ValueError("'dims' must contain integers >= 1.")

    N = len(dims)

    if not isinstance(keep, list):
        raise TypeError("'keep' must be a list.")

    if not all(isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in keep):
        raise TypeError("'keep' must contain only integers.")

    if any(s < 0 or s >= N for s in keep):
        raise ValueError("'keep' contains invalid indices.")

    if not np.issubdtype(psi.dtype, np.number):
        raise TypeError("'psi' must contain only numbers.")

    if psi.ndim not in (1, 2):
        raise ValueError("'psi' must be 1D or 2D.")

    if not np.all(np.isfinite(psi)):
        raise ValueError("'psi' contains NaN or infinite values.")

    psi = psi.reshape(-1)

    D = int(np.prod(dims))

    if psi.size != D:
        raise ValueError(f"'psi' must have {D} elements.")
    
    trace = [i for i in range(N) if i not in keep]
    d_keep = int(np.prod([dims[k] for k in keep])) if keep else 1
    M = psi.reshape(dims).transpose(keep + trace).reshape(d_keep, -1)

    return M @ M.conj().T



def main():
    #------------------------
    # TENSOR PRODUCT
    #------------------------
    print("\n========== TENSOR PRODUCT ==========")
    try:

        A1 = np.random.rand(3,3)
        B1 = np.random.rand(3,3)

        assert(np.allclose(my_kron(A1, B1), np.kron(A1, B1)))

        A2 = np.random.rand(4,4)
        B2 = np.random.rand(4,4)

        assert(np.allclose(my_kron(A2, B2), np.kron(A2, B2)))

        print(f"The tensor product of {A1} and {A2} is {my_kron(A1, B1)}")
    except (TypeError, ValueError) as e:
        print(f"Input error: {e}")
    except AssertionError as e:
        print(f"Numerical check failed: {e}")

    #-----------------------
    # MANY BODY STATES 
    #-----------------------
    print("\n========== MANY BODY STATES ==========")

    try:
        v0 = np.array([1, 0], dtype=complex)
        v1 = np.array([0, 1], dtype=complex)

        v000 = my_kron(v0, my_kron(v0, v0)).reshape(-1)
        v111 = my_kron(v1, my_kron(v1, v1)).reshape(-1)

        ghz = (v000 + v111) / np.sqrt(2)
        rho_ghz = outer_product(ghz, ghz)
        rho_mix = 0.5 * outer_product(v000, v000) 
        + 0.5 * outer_product(v111, v111)

        assert np.allclose(rho_ghz, np.outer(ghz, ghz.conj()))
        assert np.allclose(rho_mix, 0.5 * np.outer(v000, v000.conj()) + 0.5 * np.outer(v111, v111.conj()))

        print(f"The density matrix of the GHZ state is:\n{rho_ghz}\n")
        print(f"The density matrix of the mixed state is:\n{rho_mix}\n")
 
        print(f"Is rho_ghz a valid density matrix? {check_density_matrix(rho_ghz)}")
        print(f"Is rho_mix a valid density matrix? {check_density_matrix(rho_mix)}")
 
        print(f"The purity of rho_ghz is {purity(rho_ghz)}")
        print(f"The purity of rho_mix is {purity(rho_mix)}")


    except (TypeError, ValueError) as e:
        print(f"Input error in many-body states: {e}")

    except AssertionError:
        print("Numerical check failed in many-body states.")

    #----------------------
    # GHZ vs. the mixed state 
    #----------------------

    print("\n========== GHZ VS MIXED STATE ==========")
    try:

        i = 0
        j = 1 
        k = 2

        sigma_z_i = embed_operator(sigma_z, [i], 3)
        sigma_z_j = embed_operator(sigma_z, [j], 3)
        sigma_x_i = embed_operator(sigma_x, [i], 3)
        sigma_x_j = embed_operator(sigma_x, [j], 3)
        sigma_x_k = embed_operator(sigma_x, [k], 3)

        sigma_z_i_ghz = np.trace(sigma_z_i @ rho_ghz)
        sigma_z_i_mix = np.trace(sigma_z_i @ rho_mix)
        print(f"<sigma_z^i> GHZ = {sigma_z_i_ghz}")
        print(f"<sigma_z^i> MIX =  {sigma_z_i_mix}")

        sigma_x_j_ghz = np.trace(sigma_x_j @ rho_ghz)
        sigma_x_j_mix = np.trace(sigma_x_j @ rho_mix)
        print(f"<sigma_x^j> GHZ = {sigma_x_j_ghz}")
        print(f"<sigma_x^j> MIX = {sigma_x_j_mix}")

        sigma_z_ij_ghz = np.trace(sigma_z_i @ sigma_z_j @ rho_ghz)
        sigma_z_ij_mix = np.trace(sigma_z_i @ sigma_z_j @ rho_mix)
        print(f"<sigma_z^i sigma_z^j> GHZ = {sigma_z_ij_ghz}")
        print(f"<sigma_z^i sigma_z^j> MIX =  {sigma_z_ij_mix}")

        sigma_x_ijk_ghz = np.trace(sigma_x_i @ sigma_x_j @ sigma_x_k @ rho_ghz)
        sigma_x_ijk_mix = np.trace(sigma_x_i @ sigma_x_j @ sigma_x_k @ rho_mix)
        print(f"<sigma_x^i sigma_x^j sigma_x^k> GHZ = {sigma_x_ijk_ghz}")
        print(f"<sigma_x^i sigma_x^j sigma_x^k> MIX = {sigma_x_ijk_mix}")

    except (TypeError, ValueError) as e:
        print(f"Input error in many-body states: {e}")

    #--------------------
    # PARTIAL TRACE 
    #--------------------

    print("\n========== PARTIAL TRACE ==========")

    try:
        i = 0
        j = 1 

        print(f"The single site reduced matrix of rho_ghz at the site {i} is {partial_trace(rho_ghz, [i], [2, 2, 2])}")
        print(f"The single site reduced matrix of rho_mix at the site {i} is {partial_trace(rho_mix, [i], [2, 2, 2])}")

        print(f"The two-site reduced matrix of rho_ghz at the site {i} and {j} is {partial_trace(rho_ghz, [i, j], [2, 2, 2])}")
        print(f"The two-site reduced matrix of rho_mix at the site {i} and {j} is {partial_trace(rho_mix, [i, j], [2, 2, 2])}")

    except (TypeError, ValueError) as e:
            print(f"Input error in many-body states: {e}")

    #-------------------------------------------
    # TWO ROUTES TO THE SAME EXPECTATION VALUE 
    #-------------------------------------------

    print("\n========== FULL VS REDUCED EXPECTATION ==========")

    try:
        op_zz = my_kron(sigma_z, sigma_z)
        print(f"Are the two methods equal, with single site and rho_ghz? {np.isclose(expect_full(rho_ghz, sigma_z, [i], 3), expect_reduced(rho_ghz, sigma_z, [i], [2, 2, 2]))}")
        print(f"Are the two methods equal, with single site and rho_mix? {np.isclose(expect_full(rho_mix, sigma_z, [i], 3), expect_reduced(rho_mix, sigma_z, [i], [2, 2, 2]))}")
        print(f"Are the two methods equal, with two sites and rho_ghz? {np.isclose(expect_full(rho_ghz, sigma_z, [i, j], 3), expect_reduced(rho_ghz, op_zz, [i, j], [2, 2, 2]))}")
        print(f"Are the two methods equal, with two sites and rho_mix? {np.isclose(expect_full(rho_mix, sigma_z, [i, j], 3), expect_reduced(rho_mix, op_zz, [i, j], [2, 2, 2]))}")
    except (TypeError, ValueError) as e:
        print(f"Input error in full vs reduced expectation: {e}")

    except (TypeError, ValueError) as e:
            print(f"Input error in many-body states: {e}")

    #-------------------------------------
    # TIMING COMPARISON
    #-------------------------------------

    print("\n========== PARTIAL TRACE TIMING ==========")
 
    try:
        N_big = 10
        dims_big = [2] * N_big
        keep = [0, 1]
 
        # Random normalized pure state and its density matrix
        psi_big = np.random.rand(2**N_big) + 1j * np.random.rand(2**N_big)
        psi_big = psi_big / np.linalg.norm(psi_big)
        rho_big = outer_product(psi_big, psi_big)
 
        t0 = time.perf_counter()
        r_rho = partial_trace(rho_big, keep, dims_big)
        t_rho = time.perf_counter() - t0
 
        t0 = time.perf_counter()
        r_psi = partial_trace_pure(psi_big, keep, dims_big)
        t_psi = time.perf_counter() - t0
 
        print(f"N = {N_big}, keep = {keep}")
        print(f"partial_trace:      {t_rho:.3e} s")
        print(f"partial_trace_pure: {t_psi:.3e} s")
        print(f"speedup: {t_rho / t_psi:.1f}")
 
    except (TypeError, ValueError) as e:
        print(f"Input error in timing: {e}")



if __name__ == "__main__":
    main()