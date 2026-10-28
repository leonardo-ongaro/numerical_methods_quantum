import numpy as np

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
            C[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B

    return C 

def outer_product(A, B):
    C = np.zeros((len(A), len(B)))

    for i in range(len(A)):
        for j in range(len(B)):
            C[i, j] = A[i] * B[j]

    return C

def embed_operator(op, sites, N, D):
    I = np.eye(D)
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


def main():
    v0 = np.array([1, 0])
    v1 = np.array([0,1])
    m0 = outer_product(v0, v0)
    m1 = outer_product(v1, v1)

    ghz = 1/np.sqrt(2) (my_kron(v0, my_kron(v0, v0)) + my_kron(v1, my_kron(v1, v1)) )