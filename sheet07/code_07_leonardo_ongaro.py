import numpy as np 

def wave_function(x):
    return np.pi ** (-1/4) * np.exp(-x**2/2)

def momentum_wave_function(p):
    return np.pi ** (-1/4) * np.exp(-p**2/2)


def main():
    N = 100
    x_min = -10
    x_max = 10
    dx = (x_max - x_min) / N
    dk = 2*np.pi/(N*dx)

    x = np.arange(x_min, x_max, dx)
    k = 2*np.pi*np.fft.fftfreq(N, d=dx)

    wf = wave_function(x) * np.sqrt(dx)
    norm = np.linalg.norm(wf)

    mom_wf = momentum_wave_function(k) * np.sqrt(dk)
    mom_norm = np.linalg.norm(mom_wf)

    print(f"Are the two norms close?: {np.isclose(norm, mom_norm)}")

    gs_energy = 0.5 * np.sum(k**2 * np.abs(mom_wf)**2)
    print(f"The ground state energy is omega/4?: {np.isclose(gs_energy, 1/4)}")


if __name__ == "__main__":
    main()