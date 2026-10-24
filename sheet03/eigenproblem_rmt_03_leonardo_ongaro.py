import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def P(s, a, alpha, b, beta):
    return a * s**alpha * np.exp(-b * s**beta)


def spacings(eigenvalues):
    d = np.diff(eigenvalues)
    s = d / np.mean(d)
    return s[:-2]


def fit_and_plot(s, title, p0=(1, 1, 1, 2), bins=200):
    counts, edges = np.histogram(s, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    popt, pcov = curve_fit(
        P, centers, counts, p0=p0,
        bounds=([0, 0, 0, 0.1], [np.inf, 10, np.inf, 10]),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    names = ["a", "alpha", "b", "beta"]
    print(f"\n--- {title} ---")
    for n, v, e in zip(names, popt, perr):
        print(f"{n:6s} = {v:.4f} ± {e:.4f}")

    x = np.linspace(centers.min(), centers.max(), 500)
    plt.figure()
    plt.hist(s, bins=bins, density=True, alpha=0.6, label="dati")
    plt.plot(x, P(x, *popt), "r-", lw=2,
             label=f"fit: α={popt[1]:.2f}, β={popt[3]:.2f}")
    plt.xlabel("s")
    plt.ylabel("P(s)")
    plt.title(title)
    plt.legend()
    plt.show()
    return popt, perr


if __name__ == "__main__":
    N = 5000

    A = np.random.rand(N, N)
    A = (A + A.T) / 2
    ev1 = eigh(A, eigvals_only=True)
    s1 = spacings(ev1)
    fit_and_plot(s1, "Real symmetric", p0=(1, 1, 1, 2))

    B = np.diag(np.random.rand(N))
    ev2 = eigh(B, eigvals_only=True)
    s2 = spacings(ev2)
    fit_and_plot(s2, "Diagonal", p0=(1, 0.1, 1, 1))

    X = np.random.rand(N, N) + 1j * np.random.rand(N, N)
    C = (X + X.conj().T) / 2
    ev3 = eigh(C, eigvals_only=True)
    s3 = spacings(ev3)
    fit_and_plot(s3, "Hermitian", p0=(1, 2, 1, 2))