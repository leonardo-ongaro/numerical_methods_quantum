import numpy as np
import time 

def matmul(A, B):
    """Matrix-matrix product using three nested loops."""
    n, k_dim, m = A.shape[0], A.shape[1], B.shape[1]
    C = np.zeros((n,m))

    for i in range(n):
        for j in range (m):
            for k in range(k_dim):
                C[i][j] += A[i][k] * B[k][j]

    return C

def model(x, c, p):
    return c * x ** p

if __name__ == "__main__":
    # SANITY CHECK
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    C1 = matmul(A,B)
    C2 = A @ B

    print("Matrix-matrix multiplication")
    print(f"Computet with nested loops: {C1}") 
    print(f"Computed with build-in matmul: {C2}")
    print(f"Are the two results close? {np.allclose(C1, C2)}")

    # SCALING OF THE MATRIX-MATRIX MULTIPLICATION
    N_min = 10
    N_max = 120

    dt_matmul = []

    for i in range(N_min, N_max):
        A = np.random.rand(i, i)
        B = np.random.rand(i, i)

        t_i = time.perf_counter()
        C = matmul(A, B)
        t_f = time.perf_counter()
        dt_matmul +=  [t_f - t_i]

    Ns = np.arange(N_min, N_max)
    dt_matmul = np.array(dt_matmul)

    data_csv = np.column_stack((Ns, dt_matmul))
    np.savetxt("matmul_times.csv", data_csv, delimiter=",", fmt="%g")

    

    N_max = 2_500
    dt_numpy = []

    for i in range(N_min, N_max):
        print(i)
        A = np.random.rand(i, i)
        B = np.random.rand(i, i)

        t_i = time.perf_counter()
        C = A @ B
        t_f = time.perf_counter()
        dt_numpy += [t_f - t_i] 

    Ns = np.arange(N_min, N_max)
    dt_numpy = np.array(dt_numpy)

    data_csv = np.column_stack((Ns, dt_numpy))
    np.savetxt("numpy_times.csv", data_csv, delimiter=",", fmt="%g")

    
