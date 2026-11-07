import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags_array 
from scipy.sparse.linalg import eigsh 
import matplotlib.pyplot as plt

def harm_osc_solver(N, L, omega, k):
    delta_x = 2 * L / (N-1)

    # Upper diagonal 
    ud = -1/(2*delta_x**2) * np.ones(N-1)
    # Lower diagonal 
    ld = -1/(2*delta_x**2) * np.ones(N-1)
    # Main diagonal 
    md = 1/(delta_x**2) * np.ones(N)
    for i in range(N):
        x = -L + i * delta_x
        md[i] += omega**2/2*x**2
    # hamiltonian 
    H = np.diag(md, 0) + np.diag(ud, 1) + np.diag(ld, -1)
    # eigenvalues and eigenvectros
    return eigh(H, subset_by_index=[0,k-1])

def errors_ho(eigvals, omega, k):
    en_spectrum = np.array([omega*(n + 0.5) for n in range(k)])
    errors = np.abs(eigvals - en_spectrum)  

    return errors

def quartic_oscillator_solver(N, L, omega, k):
    delta_x = 2 * L / (N-1)

    # Upper diagonal 
    ud = -1/(2*delta_x**2) * np.ones(N-1)
    # Lower diagonal 
    ld = -1/(2*delta_x**2) * np.ones(N-1)
    # Main diagonal 
    md = 1/(delta_x**2) * np.ones(N)
    for i in range(N):
        x = -L + i * delta_x
        md[i] += omega**2/2*x**4
    # hamiltonian 
    H = np.diag(md, 0) + np.diag(ud, 1) + np.diag(ld, -1)
    # eigenvalues and eigenvectros
    return eigh(H, subset_by_index=[0,k-1])

def harm_osc_solver2(N, L, omega):
    delta_x = 2 * L / (N-1)

    # Upper diagonal 
    ud = -1/(2*delta_x**2) * np.ones(N-1)
    # Lower diagonal 
    ld = -1/(2*delta_x**2) * np.ones(N-1)
    # Main diagonal 
    md = 1/(delta_x**2) * np.ones(N)
    for i in range(N):
        x = -L + i * delta_x
        md[i] += omega**2/2*x**2
    # hamiltonian 
    H = diags_array([md, ud, ld], offsets=[-1, 0, 1])
    # eigenvalues and eigenvectros
    return eigsh(H)

def main():
    N = 1000
    L = 100
    omega = 1
    k = 10

    eigvals, eigvectors = harm_osc_solver(N, L, omega, k)
    print(f"\nThe first {k} eigenvalues are:", eigvals)
    print(f"\nThe eigenvectors are:", eigvectors)
    
    errors = errors_ho(eigvals, omega, k)
    plt.plot(range(k), errors)

    plt.xlabel("Eigenvalue index")
    plt.ylabel("Error of the eigenvalue")
    plt.title("Errors vs index of the eigenvalue")
    plt.grid(True)
    plt.show()

    L = 100
    Nmin = 10
    Nmax = 2000
    step = 10
    Ns = range(Nmin, Nmax, step)
    errors = np.zeros(len(Ns))
    for N in Ns:
        eigvals, _ = harm_osc_solver(N, L, omega, 1)
        errors[int((N-Nmin)/step)] = errors_ho(eigvals, omega, 1)[0]

    plt.loglog(Ns, errors)
    plt.xlabel("N")
    plt.ylabel("Error of the eigenvalue")
    plt.title("Errors vs N")
    plt.grid(True)
    plt.show()

    N = 1000
    Lmin = 1
    Lmax = 2001
    step = 10
    Ls = range(Lmin, Lmax, step)
    errors = np.zeros(len(Ls))
    for L in Ls:
        eigvals, _ = harm_osc_solver(N, L, omega, 1)
        errors[int((L-Lmin)/step)] = errors_ho(eigvals, omega, 1)[0]

    plt.loglog(Ls, errors)
    plt.xlabel("L")
    plt.ylabel("Error of the eigenvalue")
    plt.title("Errors vs L")
    plt.grid(True)
    plt.show()

    L = 100
    Nmin = 10
    Nmax = 2000
    step = 10
    Ns = range(Nmin, Nmax, step)
    errors = np.zeros(len(Ns))
    for N in Ns:
        eigvals, _ = harm_osc_solver2(N, L, omega)
        errors[int((N-Nmin)/step)] = errors_ho(eigvals, omega, 1)[0]
    
    plt.loglog(Ns, errors)
    plt.xlabel("N")
    plt.ylabel("Error of the eigenvalue")
    plt.title("Errors vs N")
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()