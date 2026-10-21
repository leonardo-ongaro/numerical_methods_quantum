import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def t(N, c, p):
    return c * N ** p

if __name__ == "__main__":
    data_matmul = np.loadtxt("matmul_times.csv", delimiter=",")
    Nsm = data_matmul[:, 0]
    dt_matmul = data_matmul[:, 1]

    popt, pcov = curve_fit(t, Nsm, dt_matmul)

    c, p = popt 
    print(f"Parameters for matrix-matrix multiplication with nested loops: {c}, {p}")
    dt_matmul_fit = t(Nsm, c, p)

    plt.plot(Nsm, dt_matmul)
    plt.xlabel("Dimension of the matrix")
    plt.ylabel("Time for the matrix - matrix multiplication")
    plt.title("Matrix-matrix multiplication with nested for loops")
    plt.grid(True)
    plt.show()

    data_numpy = np.loadtxt("numpy_times.csv", delimiter=",")
    Nsn = data_numpy[:, 0]
    dt_numpy = data_numpy[:, 1]
    
    popt, pcov = curve_fit(t, Nsn, dt_numpy)

    c, p = popt 
    print(f"Parameters for numpy matmul multiplication: {c}, {p}")
    dt_numpy_fit = t(Nsn, c, p)

    plt.plot(Nsn, dt_numpy)
    plt.xlabel("Dimension of the matrix")
    plt.ylabel("Time for the matrix - matrix multiplication")
    plt.title("Matrix-matrix multiplication with numpy matmul")
    plt.grid(True)
    plt.show()

    plt.plot(Nsm, dt_matmul_fit, label="Matmul fit")
    plt.plot(Nsn, dt_numpy_fit, label="Numpy fit")
    plt.xlabel("Dimension of the matrix")
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("Time for the matrix - matrix multiplication")
    plt.title("Comparison of the two fits")
    plt.legend()
    plt.grid(True)
    plt.show()