"""
    Contents 
    --------
    - harm_osc_solver: compute the eigenvalues and eigenvectors of the hamiltonian of the harmonic oscillator
    - errors_ho: compute the errors in the numerical eigenvalues by comparison with analytical eigenvalues
    - quartic_oscillator_solver: compute the eigenvalues and eigenvectors of the hamiltonian H = p^2/2 + 1/2*lambda x^4
    - harm_osc_solver2: optimized version of harm_osc_solver with sparse matrices
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags_array 
from scipy.sparse.linalg import eigsh 
import matplotlib.pyplot as plt
import time

def harm_osc_solver(N, L, omega, k):
    """
        Compute the eigenvalues and eigenvectors of the hamiltonian of the harmonic oscillator H = p^2/2 + 1/2*omega^2*x^2.

        Parameters
        ----------
        N: integer
            Number of points in the grid
        L: float or integer 
            Length of the box (x_i in [-L, L])
        omega: float or integer 
            Frequency of the harmonic oscillator 
        k: integer 
            The function computes the first k eigenvalues  

        Returns 
        -------
        tuple of vector and matrix 
            Vector of eigenvalues and matrix of eigenvectors 
        
        Raises
        ------
        TypeError
            If N and k are not integers and if L and omega are not non-complex numbers
        ValueError
            If N, L, omega and k are negative and if k>N
    """
    # INPUT CHECKS 
    if not isinstance(N, (np.integer, int)):
        raise TypeError("N must be an integer")
    if not isinstance(k, (np.integer, int)):
        raise TypeError("k must be an integer")
    if not isinstance(L, (np.integer, int, float, np.floating)):
        raise TypeError("L must be an integer or a float")
    if not isinstance(omega, (np.integer, int, float, np.floating)):
        raise TypeError("omega must be an integer or a float")

    if N <= 0:
        raise ValueError("N must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if omega < 0:
        raise ValueError("omega must be non negative")
    if k < N and k > N:
        raise ValueError("k must be positive and lesser than N")

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

def errors_ho(eigvals, omega):
    """
        Compute the errors of the numerical eigenvalues by comparison with analytical eigenvalues

        Parameters
        ----------
        eigvals: np,ndarray of floats
            Numerical eigenvalues of the harmonic oscillator
        omega: float or integer 
            Frequency of the harmonic oscillator 

        Returns 
        -------
        np.ndarray 
            Vector with the errors
        
        Raises
        ------
        TypeError
            If omega is not a non-complex number and if eigvals if not a np.ndarray of floats
        ValueError
            If omega is negative and if there are negative eigenvalues
    """
    # INPUT CHECKS 
    if not isinstance(eigvals, np.ndarray):
        raise TypeError("eigvals must be an array")
    if not np.issubdtype(eigvals.dtype, np.floating):
        raise TypeError("eigvals must contain floating numbers")
    if not isinstance(omega, (np.integer, int, float, np.floating)):
        raise TypeError("omega must be an integer or a float")

    if (eigvals <= 0).any():
        raise ValueError("N must be positive")
    if omega < 0:
        raise ValueError("omega must be non negative")
    k = len(eigvals)
    en_spectrum = np.array([omega*(n + 0.5) for n in range(k)])
    errors = np.abs(eigvals - en_spectrum)  

    return errors

def quartic_oscillator_solver(N, L, lam, k):
    """
        Compute the eigenvalues and eigenvectors of the hamiltonian of the quartic oscillator H = p^2/2 + lambda*x^4.

        Parameters
        ----------
        N: integer
            Number of points in the grid
        L: float or integer 
            Length of the box (x_i in [-L, L])
        lam: float or integer 
            Oscillator parameter 
        k: integer 
            The function computes the first k eigenvalues  

        Returns 
        -------
        tuple of vector and matrix 
            Vector of eigenvalues and matrix of eigenvectors 
        
        Raises
        ------
        TypeError
            If N and k are not integers and if L and lam are not non-complex numbers
        ValueError
            If N, L, lam and k are negative and if k>N
    """
    # INPUT CHECKS 
    if not isinstance(N, (np.integer, int)):
        raise TypeError("N must be an integer")
    if not isinstance(k, (np.integer, int)):
        raise TypeError("k must be an integer")
    if not isinstance(L, (np.integer, int, float, np.floating)):
        raise TypeError("L must be an integer or a float")
    if not isinstance(lam, (np.integer, int, float, np.floating)):
        raise TypeError("lam must be an integer or a float")

    if N <= 0:
        raise ValueError("N must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if lam < 0:
        raise ValueError("lam must be non negative")
    if k < 0 and k > N:
        raise ValueError("k must be positive and lesser than N")
    delta_x = 2 * L / (N-1)

    # Upper diagonal 
    ud = -1/(2*delta_x**2) * np.ones(N-1)
    # Lower diagonal 
    ld = -1/(2*delta_x**2) * np.ones(N-1)
    # Main diagonal 
    md = 1/(delta_x**2) * np.ones(N)
    for i in range(N):
        x = -L + i * delta_x
        md[i] += lam**2*x**4
    # hamiltonian 
    H = np.diag(md, 0) + np.diag(ud, 1) + np.diag(ld, -1)
    # eigenvalues and eigenvectros
    return eigh(H, subset_by_index=[0,k-1])

def harm_osc_solver2(N, L, omega, k):
    """
        Compute the eigenvalues and eigenvectors of the hamiltonian of the harmonic oscillator H = p^2/2 + 1/2*omega^2*x^2 with optimization for sparse matrices.

        Parameters
        ----------
        N: integer
            Number of points in the grid
        L: float or integer 
            Length of the box (x_i in [-L, L])
        omega: float or integer 
            Frequency of the harmonic oscillator 
        k: integer 
            The function computes the first k eigenvalues  

        Returns 
        -------
        tuple of vector and matrix 
            Vector of eigenvalues and matrix of eigenvectors 
        
        Raises
        ------
        TypeError
            If N and k are not integers and if L and omega are not non-complex numbers
        ValueError
            If N, L, omega and k are negative and if k>N
    """
    # INPUT CHECKS 
    if not isinstance(N, (np.integer, int)):
        raise TypeError("N must be an integer")
    if not isinstance(k, (np.integer, int)):
        raise TypeError("k must be an integer")
    if not isinstance(L, (np.integer, int, float, np.floating)):
        raise TypeError("L must be an integer or a float")
    if not isinstance(omega, (np.integer, int, float, np.floating)):
        raise TypeError("omega must be an integer or a float")

    if N <= 0:
        raise ValueError("N must be positive")
    if L <= 0:
        raise ValueError("L must be positive")
    if omega < 0:
        raise ValueError("omega must be non negative")
    if k < N and k > N:
        raise ValueError("k must be positive and lesser than N")
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
    H = diags_array([ld, md, ud], offsets=[-1, 0, 1])
    # eigenvalues and eigenvectros
    return eigsh(H, k, sigma=0) # sigma=0 for the smallest eigenvalue

def main():
    # EIGENVALUES AND EIGENVECTORS
    omega = 1
    try:
        N = 1000
        L = 10
        k = 10

        eigvals, eigvectors = harm_osc_solver(N, L, omega, k)
        print(f"\nThe first {k} eigenvalues are:", eigvals)
        print(f"\nThe eigenvectors are:", eigvectors)
    
        errors = errors_ho(eigvals, omega)
        plt.plot(range(k), errors)

        plt.xlabel("Eigenvalue index")
        plt.ylabel("Error of the eigenvalue")
        plt.title("Errors vs index of the eigenvalue")
        plt.grid(True)
        plt.show()

    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")

    # CONVERGENCE STUDY (FIXED L AND VARYING N)
    try:
        L = 10
        Nmin = 10
        Nmax = 2000
        step = 10
        Ns = range(Nmin, Nmax, step)
        errors = np.zeros(len(Ns))
        times = np.zeros(len(Ns))
        for N in Ns:
            start_time = time.perf_counter()
            eigvals, _ = harm_osc_solver(N, L, omega, 1)
            errors[int((N-Nmin)/step)] = errors_ho(eigvals, omega)[0]
            times[int((N-Nmin)/step)] = time.perf_counter() - start_time

        plt.loglog(Ns, errors)
        plt.xlabel("N")
        plt.ylabel("Error of the eigenvalue")
        plt.title("Errors vs N (L=10)")
        plt.grid(True)
        plt.show()

        plt.loglog(Ns, times)
        plt.xlabel("N")
        plt.ylabel("Times")
        plt.title("Times vs N")
        plt.grid(True)
        plt.show()

    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")

    # CONVERGENCE STUDY (FIXED N AND VARYING L)
    try:
        N = 1000
        Lmin = 1
        Lmax = 601
        step = 10
        Ls = range(Lmin, Lmax, step)
        errors = np.zeros(len(Ls))
        for L in Ls:
            eigvals, _ = harm_osc_solver(N, L, omega, 1)
            errors[int((L-Lmin)/step)] = errors_ho(eigvals, omega)[0]

        plt.loglog(Ls, errors)
        plt.xlabel("L")
        plt.ylabel("Error of the eigenvalue")
        plt.title("Errors vs L (N=1000)")
        plt.grid(True)
        plt.show()

    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")

    # FLEXIBILITY CHECK
    try:
        N = 1000
        L = 100
        lam = 1
        k = 10

        eigvals, eigvectors = quartic_oscillator_solver(N, L, lam, k)
        print(f"\nThe first {k} eigenvalues are:", eigvals)
        print(f"\nThe eigenvectors are:", eigvectors)
    

    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")

    # OPTIMIZATION
    try:
        L = 10
        Nmin = 10
        Nmax = 2000
        step = 10
        Ns = range(Nmin, Nmax, step)
        times = np.zeros(len(Ns))
        errors = np.zeros(len(Ns))
        for N in Ns:
            start_time = time.perf_counter()
            eigvals, _ = harm_osc_solver2(N, L, omega, 1)
            errors[int((N-Nmin)/step)] = errors_ho(eigvals, omega)[0]
            times[int((N-Nmin)/step)] = time.perf_counter()-start_time
    
        plt.loglog(Ns, errors)
        plt.xlabel("N")
        plt.ylabel("Error of the eigenvalue")
        plt.title("Errors vs N")
        plt.grid(True)
        plt.show()

        plt.loglog(Ns, times)
        plt.xlabel("N")
        plt.ylabel("Times")
        plt.title("Times vs N with optimized code")
        plt.grid(True)
        plt.show()

    except ValueError as e:
        print(f"Value error: {e}")
    except TypeError as e:
        print(f"Type error: {e}")



if __name__ == "__main__":
    main()