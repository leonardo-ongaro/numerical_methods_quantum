import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def matmul(A, B):
    """
    Multiply two matrices using three nested loops.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix of shape (n, k).
    B : numpy.ndarray
        Matrix of shape (k, m).

    Returns
    -------
    numpy.ndarray
        Product matrix of shape (n, m).

    Raises
    ------
    TypeError
        If A or B is not a NumPy array.
    ValueError
        If the matrices are not two-dimensional or their
        dimensions are incompatible.
    """

    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("A and B must be numpy arrays")

    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be two-dimensional")

    # check if the number of columns of A is equal to the number of rows of B
    if A.shape[1] != B.shape[0]:
        raise ValueError("Incompatible dimensions: A.shape[1] must be equal to B.shape[0]")

    if not np.issubdtype(A.dtype, np.number) or not np.issubdtype(B.dtype, np.number):
        raise TypeError("A and B must contain numeric values.")

    n, k_dim = A.shape # dimensions of the matrix A
    m = B.shape[1] # number of columns of B

    C = np.zeros((n, m), dtype=np.result_type(A.dtype, B.dtype))

    for i in range(n):
        for j in range(m):
            for k in range(k_dim):
                C[i, j] += A[i, k] * B[k, j]

    return C


def sanity_check():
    """
    Compute the two products.

    Returns
    ------
    numpy.ndarray, numpy.ndarray
        Matrix multiplication with custom function and matrix multiplication with numpy.matmul
    """
    # two random matrices
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    C_custom = matmul(A, B)
    C_numpy = A @ B

    return C_custom, C_numpy


def model(N, c, p):
    """
    Evaluate the power-law model t(N) = c * N**p.

    Parameters
    ----------
    N : int, float or numpy.ndarray
        Positive matrix dimension(s).
    c : float
        Positive multiplicative coefficient.
    p : float
        Exponent.

    Returns
    -------
    float or numpy.ndarray
        Predicted execution time(s).

    Raises
    ------
    TypeError
        If  c and p are not numbers 
    """

    if not isinstance(c, np.number) or not isinstance(p, np.number):
        raise TypeError("C and p must be numbers")

    return c * N ** p 

def measure_times(N_min, N_max, method):
    """
    Measure matrix multiplication times for increasing dimensions.

    Parameters
    ----------
    N_min : int
        Minimum matrix dimension, inclusive.
    N_max : int
        Maximum matrix dimension, exclusive.
    method : str
        Either 'loops' or 'numpy'.

    Returns
    -------
    tuple of numpy.ndarray
        Matrix dimensions and corresponding execution times.

    Raises
    ------
    TypeError
        If the bounds are not integers or method is not a string.
    ValueError
        If the bounds are invalid or method is unsupported.
    """
    if not isinstance(N_min, (int, np.integer)) or not isinstance(N_max, (int, np.integer)):
        raise TypeError("N_min and N_max must be integers")

    if N_min <= 0 or N_max <= N_min:
        raise ValueError("Bounds must satisfy 0 < N_min < N_max")

    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    if method not in ("loops", "numpy"):
        raise ValueError("method must be 'loops' or 'numpy'.")

    Ns = np.arange(N_min, N_max)
    times = np.zeros(len(Ns))


    for index, N in enumerate(Ns): # it is needed to have both the value of N (between N_min and N_max) and the step from 0 to N_max - N_min
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)

        start = time.perf_counter() 

        if method == "loops":
            C = matmul(A, B)
        else:
            C = A @ B

        end = time.perf_counter()

        times[index] = end - start


    return Ns, times

def save_times(filename, Ns, times):
    """
    Save matrix dimensions and execution times to a CSV file.

    Parameters
    ----------
    filename : str
        Output file path.
    Ns : numpy.ndarray
        One-dimensional array of matrix dimensions.
    times : numpy.ndarray
        One-dimensional array of execution times.

    Raises
    ------
    TypeError
        If the filename or arrays have invalid types.
    ValueError
        If the arrays are not one-dimensional, have different
        lengths, or contain invalid values.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")

    if not isinstance(Ns, np.ndarray) or not isinstance(times, np.ndarray):
        raise TypeError("Ns and times must be NumPy arrays.")

    if Ns.ndim != 1 or times.ndim != 1:
        raise ValueError("Ns and times must be one-dimensional.")

    if np.any(Ns <= 0):
        raise ValueError("Ns must contain positive finite values.")

    data = np.column_stack((Ns, times)) # convert the two arrays in one data with two columns
    np.savetxt(filename, data, delimiter=",", fmt="%g") # save the CSV

    print(f"Data saved to {filename}")

def load_times(filename):
    """
    Load and validate dimensions and execution times from a CSV file.

    Parameters
    ----------
    filename : str
        Input CSV file path.

    Returns
    -------
    tuple of numpy.ndarray
        Matrix dimensions and execution times.

    Raises
    ------
    TypeError
        If filename is not a string.
    ValueError
        If the file does not contain two valid columns.
    OSError
        If the file cannot be opened.
    """
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")

    data = np.loadtxt(filename, delimiter=",") # load the CSV file

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("The CSV file must contain exactly two columns.")

    Ns = data[:, 0]
    times = data[:, 1]

    return Ns, times

def fit_times(Ns, times):
    """
    Fit execution times to the model t(N) = c * N**p.

    Parameters
    ----------
    Ns : numpy.ndarray
        One-dimensional array of positive matrix dimensions.
    times : numpy.ndarray
        One-dimensional array of non-negative execution times.

    Returns
    -------
    tuple
        Fitted coefficient c, exponent p, and fitted times.

    Raises
    ------
    TypeError
        If the inputs are not NumPy arrays.
    ValueError
        If the arrays have invalid shapes or values, or if
        the fit cannot be performed.
    """
    if not isinstance(Ns, np.ndarray) or not isinstance(times, np.ndarray):
        raise TypeError("Ns and times must be NumPy arrays.")

    if Ns.ndim != 1 or times.ndim != 1:
        raise ValueError("Ns and times must be one-dimensional.")

    if len(Ns) != len(times):
        raise ValueError("Ns and times must have equal lengths.")


    popt, _ = curve_fit(model, Ns, times)

    c, p = popt
    fitted_times = model(Ns, c, p)

    return c, p, fitted_times

def plot_fit(Ns, times, fitted_times, title, label):
    """
    Plot measured execution times and their power-law fit.

    Parameters
    ----------
    Ns : numpy.ndarray
        Matrix NUMPY DATA
    times : numpy.ndarray
        Measured execution times.
    fitted_times : numpy.ndarray
        Predicted execution times.
    title : str
        Plot title.
           Legend label for the fit.

    Raises
    ------
    TypeError
        If the title or label is not a string.
    ValueError
        If the arrays have incompatible shapes or invalid values.
    """
    if not isinstance(title, str) or not isinstance(label, str):
        raise TypeError("title and label must be strings.")

    arrays = (Ns, times, fitted_times)

    if any(not isinstance(a, np.ndarray) for a in arrays):
        raise TypeError("Plot data must be NumPy arrays.")

    if any(a.ndim != 1 for a in arrays):
        raise ValueError("Plot data must be one-dimensional.")

    if not (len(Ns) == len(times) == len(fitted_times)):
        raise ValueError("Plot arrays must have equal lengths.")

    plt.figure()
    plt.plot(Ns, times, label="Measured time")
    plt.plot(Ns, fitted_times, label=label)
    plt.xlabel("Dimension of the matrix N")
    plt.ylabel("Execution time (s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_comparison(Nsm, fit_matmul, pm, Nsn, fit_numpy, pn):
    """
    Plot both fitted power laws on logarithmic axes.

    Parameters
    ----------
    Nsm : numpy.ndarray
        Dimensions used for the nested-loop experiment.
    fit_matmul : numpy.ndarray
        Fitted nested-loop execution times.
    pm : float
        Nested-loop fitted exponent.
    Nsn : numpy.ndarray
        Dimensions used for the NumPy experiment.
    fit_numpy : numpy.ndarray
        Fitted NumPy execution times.
    pn : float
        NumPy fitted exponent.

    Raises
    ------
    ValueError
        If the data arrays are invalid or have incompatible shapes.
    """
    if Nsm.ndim != 1 or Nsn.ndim != 1:
        raise ValueError("Dimension arrays must be one-dimensional.")

    if len(Nsm) != len(fit_matmul) or len(Nsn) != len(fit_numpy):
        raise ValueError("Dimension and fit arrays must have equal lengths.")

    plt.figure()
    plt.plot(Nsm, fit_matmul, label=f"Nested loops (p = {pm:.2f})")
    plt.plot(Nsn, fit_numpy, label=f"NumPy (p = {pn:.2f})")

    plt.xlabel("Dimension of the matrix N")
    plt.ylabel("Execution time (s)")
    plt.title("Comparison of power-law fits")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()


def main():
    try:
        # 1. SANITY CHECK
        C_custom, C_numpy = sanity_check()
        print(f"The two matrix-matrix products are close: {np.allclose(C_custom, C_numpy)}")

        # 2. MEASURE NESTED-LOOP MULTIPLICATION
        Ns_matmul, times_matmul = measure_times(N_min=10, N_max=120, method="loops")

        save_times("matmul_times.csv", Ns_matmul, times_matmul)

        # 3. MEASURE NUMPY MULTIPLICATION 
        Ns_numpy, times_numpy = measure_times(N_min=10, N_max=2000, method="numpy")

        save_times("numpy_times.csv", Ns_numpy, times_numpy)

        # 4. RELOAD THE CSV FILES
        Nsm, dt_matmul = load_times("matmul_times.csv")
        Nsn, dt_numpy = load_times("numpy_times.csv")

        # 5. FIT NESTED-LOOP DATA
        c_matmul, p_matmul, fit_matmul = fit_times(Nsm, dt_matmul)

        print("\nFIT: NESTED LOOPS")
        print(f"c = {c_matmul:.6e}")
        print(f"p = {p_matmul:.6f}")

        plot_fit(
            Nsm,
            dt_matmul,
            fit_matmul,
            "Matrix multiplication with nested loops",
            "Power-law fit"
        )

        # 6. FIT NUMPY DATA
        c_numpy, p_numpy, fit_numpy = fit_times(Nsn, dt_numpy)

        print("\nFIT: NUMPY MATMUL")
        print(f"c = {c_numpy:.6e}")
        print(f"p = {p_numpy:.6f}")

        plot_fit(Nsn, dt_numpy, fit_numpy, "Matrix multiplication with NumPy", "Power-law fit")

        # 7. COMPARE THE FITS
        plot_comparison(Nsm, fit_matmul, p_matmul, Nsn, fit_numpy, p_numpy
        )

        
        plt.show()

    except (TypeError, ValueError, AssertionError, OSError, RuntimeError, FloatingPointError) as error:
        print(f"\nError during execution: {error}")


if __name__ == "__main__":
    main()