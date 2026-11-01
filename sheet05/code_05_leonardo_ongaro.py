import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def one_sided_derivative(f, x0, h):
    return (f(x0+h)-f(x0))/h

def central_derivative(f, x0, h):
    return (f(x0+h)-f(x0-h))/(2*h)

def model(x, a, b):
    return a * x ** b

def main():
    h = 10**-2
    x0 = 1

    exact_derivative = np.exp(x0)

    print(f"Derivatives of exp(x) for x = {x0}")
    print(f"One sided derivative: {one_sided_derivative(np.exp, x0, h)}")
    print(f"Central differences derivative: {central_derivative(np.exp, x0, h)}")
    print(f"Analytical derivative: {exact_derivative}")

    N = 100
    h = np.logspace(-1, -6, N)
    one_sided_dvs = np.array([one_sided_derivative(np.exp, x0, i) for i in h])
    cen_dvs = np.array([central_derivative(np.exp, x0, i) for i in h])
    err_one_side = np.abs(one_sided_dvs - exact_derivative)
    err_cen = np.abs(cen_dvs - exact_derivative)

    popt1, pcov1 = curve_fit(model, h, err_one_side)
    a1, b1 = popt1

    popt2, pcov2 = curve_fit(model, h, err_cen)
    a2, b2 = popt2

    plt.loglog(h, err_one_side, 'o', label='One-sided derivatives')
    plt.loglog(h, err_cen, 's', label='Central derivatives')

    plt.loglog(h, model(h, a1, b1), '--', label=f'Fit one-sided derivatives: y={a1:.3e} x^{b1:.3f}')

    plt.loglog(h, model(h, a2, b2), '--', label=f'Fit centrale derivatives: y={a2:.3e} x^{b2:.3f}')

    plt.xlabel('h')
    plt.ylabel('errors')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()