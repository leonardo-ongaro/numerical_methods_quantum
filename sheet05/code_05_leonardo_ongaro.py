import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial.legendre import leggauss

def one_sided_derivative(f, x0, h):
    return (f(x0+h)-f(x0))/h

def central_derivative(f, x0, h):
    return (f(x0+h)-f(x0-h))/(2*h)

def model(x, a, b):
    return a * x ** b

def int_trap(f, a, b, N):
    h = (b-a)/N 
    sum = 0.5 * f(a) + 0.5 * f(b)
    x = a + h

    for i in range(1, N):
        sum += f(x)
        x += h 

    sum *= h 
    return sum 

def simpson_integral(f, a, b, N):

    if N % 2 != 0:
        raise ValueError("N must be even")

    h = (b-a)/N 
    sum = f(a) + f(b)
    x = a + h
    for i in range(1, N):
        if i % 2 == 0:
            sum += 2 * f(x)
        else:
            sum += 4 * f(x)
        x += h 

    sum *= h/3
    return sum 

def gauss_legendre(f, a, b, n):
    x, w = leggauss(n)                  
    t = 0.5*(b-a)*x + 0.5*(b+a)         
    return 0.5*(b-a)*np.sum(w*f(t))

def not_smooth_f(x):
    return np.abs(x-0.3)

def main():
    # DERIVATIVES
    h = 10**-2
    x0 = 1

    exact_derivative = np.exp(x0)

    print(f"Derivatives of exp(x) for x = {x0}")
    print(f"One sided derivative: {one_sided_derivative(np.exp, x0, h)}")
    print(f"Central differences derivative: {central_derivative(np.exp, x0, h)}")
    print(f"Analytical derivative: {exact_derivative}")

    N = 50
    h = np.logspace(-1, -5, N)
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
    plt.title("Error comparison")

    plt.xlabel('h')
    plt.ylabel('errors')
    plt.legend()
    plt.grid(True)
    plt.show()

    h = np.logspace(-1, -16, N)
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
    plt.title("Numerical instability")

    plt.xlabel('h')
    plt.ylabel('errors')
    plt.legend()
    plt.grid(True)
    plt.show()    

    # INTEGRALS 
    a = 0 
    b = np.pi 
    N = 100

    print(f"Integral of sin(x) between a = {a} and b = {b}")
    print(f"Trapezoidal rule: {int_trap(np.sin, a, b, N)}")
    print(f"Simpson rule: {simpson_integral(np.sin, a, b, N)}")
    print("Exact result: 2")

    N = np.geomspace(20, 2000, N)
    N = np.unique(2*np.round(N/2).astype(int))
    trap = np.array([int_trap(np.sin, a, b, i) for i in N])
    simpson = np.array([simpson_integral(np.sin, a, b, i) for i in N])
    err_trap = np.abs(trap - 2)
    err_simpson = np.abs(simpson - 2)

    popt1, pcov1 = curve_fit(model, N, err_trap)
    a1, b1 = popt1

    popt2, pcov2 = curve_fit(model, N, err_simpson)
    a2, b2 = popt2

    plt.loglog(N, err_trap, 'o', label='Trapezoidal itegrals')
    plt.loglog(N, err_simpson, 's', label='Simpson integrals')

    plt.loglog(N, model(N, a1, b1), '--', label=f'Fit trapeozoidal integrals: y={a1:.3e} x^{b1:.3f}')

    plt.loglog(N, model(N, a2, b2), '--', label=f'Fit simpson integrals: y={a2:.3e} x^{b2:.3f}')
    plt.title("Error comparison")

    plt.xlabel('N')
    plt.ylabel('errors')
    plt.legend()
    plt.grid(True)
    plt.show()

    # GAUSSIAN QUADRATURE 
    print(f"\nGaussian quadrature for the integral of sin(x) between 0 and pi")
    print(f"Integral with n = 2: {gauss_legendre(np.sin, 0, np.pi, 2)}")
    print(f"Integral with n = 4: {gauss_legendre(np.sin, 0, np.pi, 4)}")
    print(f"Integral with n = 6: {gauss_legendre(np.sin, 0, np.pi, 6)}")
    print(f"Integral with n = 8: {gauss_legendre(np.sin, 0, np.pi, 8)}")

    print(f"\nGaussian quadrature for the integral of |x-0.3| between -1 and 1")
    print(f"Exact result {1+0.3*0.3}")
    print(f"Integral with Simpson's rule: {simpson_integral(not_smooth_f, -1, 1, 100)}")
    print(f"Integral with n = 2: {gauss_legendre(not_smooth_f, -1, 1, 2)}")
    print(f"Integral with n = 4: {gauss_legendre(not_smooth_f, -1, 1, 4)}")
    print(f"Integral with n = 6: {gauss_legendre(not_smooth_f, -1, 1, 6)}")
    print(f"Integral with n = 8: {gauss_legendre(not_smooth_f, -1, 1, 8)}")

if __name__ == "__main__":
    main()