import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def P(s, a, alpha, b, beta):
    """
    Evaluate the spacing distribution P(s) = a * s**alpha * exp(-b * s**beta).

    Parameters
    ----------
    s : int, float or numpy.ndarray
        Non-negative normalized spacing(s).
    a : float
        Multiplicative coefficient.
    alpha : float
        Exponent of the power-law term.
    b : float
        Coefficient of the exponential term.
    beta : float
        Exponent of the exponential term.

    Returns
    -------
    float or numpy.ndarray
        Value(s) of the distribution at s.

    Raises
    ------
    ValueError
        If s contains negative values.
    """

    # check if s has valid values
    if np.any(s < 0):
        raise ValueError("s must contain finite non-negative values")

    return a * s**alpha * np.exp(-b * s**beta)


def spacings(eigenvalues):
    """
    Compute the nearest-neighbour spacings normalized to unit mean.

    The last two spacings are discarded.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        One-dimensional array of eigenvalues.

    Returns
    -------
    numpy.ndarray    TypeError
        If eigenvalues is not a NumPy array of numbers.
    ValueError
        If eigenvalues is not one-dimensional
    """
    # check if the input is a numpy array
    if not isinstance(eigenvalues, np.ndarray):
        raise TypeError("eigenvalues must be a numpy array")

    # check if the values are numbers
    if not np.issubdtype(eigenvalues.dtype, np.number):
        raise TypeError("eigenvalues must contain numeric values")

    # check if the array is one-dimensional and long enough
    if eigenvalues.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")


    d = np.diff(np.sort(eigenvalues)[:-1])

    s = d / np.mean(d)
    return s


def fit_and_plot(s, title, p0, bounds, bins = 80):
    """
    Fit the histogram of the spacings with P(s) and plot data and fit.

    The fit parameters and their uncertainties are printed. The figure
    is created but not shown: call plt.show() afterwards.

    Parameters
    ----------
    s : numpy.ndarray
        One-dimensional array of non-negative normalized spacings.
    title : str
        Plot title.
    bins : int, optional
        Number of histogram bins (at least 5). Default is 200.

    Returns
    -------
    numpy.ndarray
        Best-fit parameters (a, alpha, b, beta).

    Raises
    ------
    TypeError
        If s is not a NumPy array of numbers, title is not a
        string or bins is not an integer.
    ValueError
        If s is not one-dimensional, is empty, contains invalid
        values, or if bins is smaller than 5.
    RuntimeError
        If the fit does not converge.
    """
    # check if s is a numpy array of numbers
    if not isinstance(s, np.ndarray):
        raise TypeError("s must be a numpy array")

    if not np.issubdtype(s.dtype, np.number) or np.iscomplexobj(s):
        raise TypeError("s must contain real numeric values")

    # check if s is one-dimensional, non-empty and valid
    if s.ndim != 1 or s.size == 0:
        raise ValueError("s must be a non-empty one-dimensional array")

    if not np.all(np.isfinite(s)) or np.any(s < 0):
        raise ValueError("s must contain finite non-negative values")

    # check the title and the number of bins
    if not isinstance(title, str):
        raise TypeError("title must be a string")

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)):
        raise TypeError("bins must be an integer")


    counts, edges = np.histogram(s, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    p0 = [1.0, 1.0, 1.0, 1.0]

    bounds = ([1e-10, 0.0, 1e-10, 0.1], [100.0, 5.0, 100.0, 5.0])

    popt, _ = curve_fit(P, centers, counts, p0 = p0, bounds = bounds)

    print(f"\nFIT: {title.upper()}")
    names = ["a", "alpha", "b", "beta"]
    for n, v in zip(names, popt):
        print(f"{n} = {v}")

    x = np.linspace(centers.min(), centers.max(), 500)
    plt.figure()
    plt.hist(s, bins=bins, density=True, label="data")
    plt.plot(x, P(x, *popt))
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    return popt


def main():
    N = 5000
    try:

        # 1. REAL SYMMETRIC MATRIX
        A = np.random.rand(N, N)
        A = (A + A.T) / 2
        ev1, _ = eigh(A)
        s1 = spacings(ev1)
        fit_and_plot(s1, "Real symmetric", p0=[1.5, 1.0, 0.8, 2.0], bounds=([0.1, 0.0, 0.05, 0.5], [10.0, 4.0, 5.0, 4.0]))
        plt.show()
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"\nError with the real symmetric matrix: {error}")


    try:
        # 2. DIAGONAL MATRIX
        B = np.diag(np.random.rand(N))
        ev2, _ = eigh(B)
        s2 = spacings(ev2)
        fit_and_plot(s2, "Diagonal", p0=[1.0, 0.0, 1.0, 1.0], bounds=([0.1, 0.0, 0.05, 0.5], [10.0, 4.0, 5.0, 4.0]))
        plt.show()
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"\nError with the diagonal matrix: {error}")

    try:
        # 3. HERMITIAN MATRIX
        X = np.random.rand(N, N) + 1j * np.random.rand(N, N)
        C = (X + X.conj().T) / 2
        ev3, _ = eigh(C)
        s3 = spacings(ev3)
        fit_and_plot(s3, "Hermitian", p0=[3.0, 2.0, 1.3, 2.0], bounds=([0.1, 0.0, 0.05, 0.5], [20.0, 4.0, 5.0, 4.0]))
        plt.show()
    except (TypeError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"\nError with the Hermitian matrix: {error}")


if __name__ == "__main__":
    main()