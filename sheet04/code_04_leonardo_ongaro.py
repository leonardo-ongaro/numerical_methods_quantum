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

def my_kron(A, B):

    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n_A, m_A = A.shape 
    n_B, m_B = B.shape 

    C= np.zeros((n_A*n_B, m_A*m_B))

    for i in range(n_A):
        for j in range(m_A):
            C[i*n_B:(i+1)*n_B, j*m_B:(j+1)*m_B] = A[i, j] * B

    return C 

def outer_product(A, B):
    C = np.zeros((len(A), len(B)))

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = A[i] * B[j]

    return C

def embed_operator(op, sites, N):
    n, m = op.size
    I = np.eye(n)
    result = None  

    for site in range(1, N+1):
        if site in sites:
            local_op = op 
        else:
            local_op = I 

        if result == None:
            result = local_op 
        else:
            result = my_kron(result, local_op)

    return result 

def check_density_matrix(A):
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


