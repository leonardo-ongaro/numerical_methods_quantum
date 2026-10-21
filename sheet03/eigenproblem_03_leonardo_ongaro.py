import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    N = 7000
    A = np.random.rand(N, N)

    A = (A + A.T) / 2

    eigenvalues, _ = eigh(A)

    lambdas = np.diff(eigenvalues)
    lambda_mean = np.mean(lambdas)

    s = lambdas / lambda_mean
    s = s[:-2]

    plt.hist(s, bins=1000)
    plt.xlabel("Value of s")
    plt.ylabel("Frequency")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.title("Distribution")
    plt.show()
