"""
    Contents
    --------
    - one_sided_derivative: one sided derivative of f at point x0
    - central_derivative: centrale derivative of f at point x0
    - model: model function to fit
    - int_trap: integral of f between a and b by the trapezoid rule 
    - simpson_integral: integral of f between a and b by Simpson's rule
    - gauss_legendre: integral of f between a and b by Gauss-Legendre quadrature
    - not_smooth_f: function to integrate
"""
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial.legendre import leggauss

def one_sided_derivative(f, x0, h):
    """
        Evaluate the one-sided derivative f'(x)|_x0 = (f(x0+h)-f(x0))/h.

        Parameters
        ----------
        f: function
            Function to derive.
        x0: integer or float
            Point where the derivative is evaluated 
        h: non-negative integer or float 
            Grid step 

        Returns
        -------
        integer or float
            The one sided derivative at the point x0

        Raises
        ------
        TypeError
            If func is not a function or x0 and h are not integers or floating
        ValueError
            If h is negative
    """
    # INPUT CHECKS 
    if not isinstance(x0, (np.integer, np.floating)):
        raise TypeError("x0 must be an integer or a float")
    if not isinstance(h, (np.integer, np.floating)):
            raise TypeError("h must be an integer or a float")
    if h <= 0:
        raise ValueError("h must be positive")
    if not callable(f):
        raise TypeError("f must be a function")
    
    return (f(x0+h)-f(x0))/h

def central_derivative(f, x0, h):
    """
        Evaluate the central derivative f'(x)|_x0 = (f(x0+h)-f(x0-h))/(2*h)
    
        Parameters
        ----------
        f: function
            Function to derive.
        x0: integer or float
            Point where the derivative is evaluated 
        h: non-negative integer or float 
            Grid step 
    
        Returns
        -------
        integer or float
            The central derivative at the point x0
    
        Raises
        ------
        TypeError
            If func is not a function or x0 and h are not integers or floating
        ValueError
            If h is negative
    """
    # INPUT CHECKS 
    if not isinstance(x0, (np.integer, np.floating)):
        raise TypeError("x0 must be an integer or a float")
    if not isinstance(h, (np.integer, np.floating)):
            raise TypeError("h must be an integer or a float")
    if h <= 0:
        raise ValueError("h must be positive")
    if not callable(f):
        raise TypeError("f must be a function")
    
    return (f(x0+h)-f(x0-h))/(2*h)

def model(x, a, b):
    """
        Function to model the behaviour of the order of convergence

        Parameters
        ----------
        x: float
            Point in which compute the function
        a: float
            Parameter of the function
        b: float 
            Parameter of the function
        
        Returns
        -------
        float
            Value of the function in x

        Raises
        ------
        TypeError
            If x is not a number and a and b are not floats
    """
    if not isinstance(x, np.number):
        raise TypeError("x must be a number")
    if not isinstance(a, np.floating) or not isinstance(b, np.floating):
         raise TypeError("a  and b must be floating numbers")
    
    return a * x ** b

def int_trap(f, a, b, N):
    """
        Evaluate the integral with the trapezoid rule
    
        Parameters
        ----------
        f: function
            Function to integrate.
        a: integer or float
            Lower bound of the integration
        b: integer or float 
            Upper bound of the integration 
        N: non-negative integer 
            Number of steps
    
        Returns
        -------
        integer or float
            The integral of f between a and b
    
        Raises
        ------
        TypeError
            If func is not a function or a, b and h are not integers or floating
        ValueError
            If h is negative or if a >= b
    """
    # INPUT CHECKS 
    if not isinstance(a, (np.integer, np.floating)) or not isinstance(b, (np.integer, np.floating)):
        raise TypeError("a and b must be integers or floats")
    if not isinstance(N, np.integer):
            raise TypeError("N must be an integer")
    if N <= 0:
        raise ValueError("N must be positive")
    if not callable(f):
        raise TypeError("f must be a function")
    if a >= b:
        raise ValueError("a must be lesser than b")
    
    h = (b-a)/N 
    sum = 0.5 * f(a) + 0.5 * f(b) # initial and final points
    x = a + h

    for i in range(1, N):
        sum += f(x)
        x += h 

    sum *= h 
    return sum 

def simpson_integral(f, a, b, N):
    """
        Evaluate the integral with the Simpson's rule
    
        Parameters
        ----------
        f: function
            Function to integrate.
        a: integer or float
            Lower bound of the integration
        b: integer or float 
            Upper bound of the integration 
        N: non-negative integer 
            Number of steps
    
        Returns
        -------
        integer or float
            The integral of f between a and b
    
        Raises
        ------
        TypeError
            If func is not a function or a, b and h are not integers or floating
        ValueError
            If h is negative or if a >= b or if N is odd
    """
    # INPUT CHECKS 
    if not isinstance(a, (np.integer, np.floating)) or not isinstance(b, (np.integer, np.floating)):
        raise TypeError("a and b must be integers or floats")
    if not isinstance(N, np.integer):
            raise TypeError("N must be an integer")
    if N <= 0:
        raise ValueError("N must be positive")
    if not callable(f):
        raise TypeError("f must be a function")
    if a >= b:
        raise ValueError("a must be lesser than b")
    if N % 2 != 0:
        raise ValueError("N must be even")

    h = (b-a)/N 
    sum = f(a) + f(b) # initial and final points
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
    """
        Evaluate the integral with the Gauss-Legendre quadrature
    
        Parameters
        ----------
        f: function
            Function to integrate.
        a: integer or float
            Lower bound of the integration
        b: integer or float 
            Upper bound of the integration 
        N: non-negative integer 
            Number of nodes
    
        Returns
        -------
        integer or float
            The integral of f between a and b
    
        Raises
        ------
        TypeError
            If func is not a function or a, b and h are not integers or floating
        ValueError
            If h is negative or if a >= b or if n is odd
    """
    # INPUT CHECKS 
    if not isinstance(a, (np.integer, np.floating)) or not isinstance(b, (np.integer, np.floating)):
        raise TypeError("a and b must be integers or floats")
    if not isinstance(n, np.integer):
            raise TypeError("N must be an integer")
    if n <= 0:
        raise ValueError("N must be positive")
    if not callable(f):
        raise TypeError("f must be a function")
    if a >= b:
        raise ValueError("a must be lesser than b")
    if n % 2 != 0:
        raise ValueError("n must be even")
    
    x, w = leggauss(n)  # weights                
    t = 0.5*(b-a)*x + 0.5*(b+a)         
    return 0.5*(b-a)*np.sum(w*f(t))

def not_smooth_f(x):
    """
        Function to integrate

        Parameters
        ----------
        x: float
            Point in which compute the function
        
        Returns
        -------
        float
            Value of the function in x

        Raises
        ------
        TypeError
            If x is not a number
    """
    if not isinstance(x, np.number):
        raise TypeError("x must be a number")
    return np.abs(x-0.3)

def main():
    # DERIVATIVES
    h = 10**-2
    x0 = 1

    exact_derivative = np.exp(x0)
    try:
        print(f"Derivatives of exp(x) for x = {x0}")
        print(f"One sided derivative: {one_sided_derivative(np.exp, x0, h)}")
        print(f"Central differences derivative: {central_derivative(np.exp, x0, h)}")
        print(f"Analytical derivative: {exact_derivative}")
    except TypeError as e:
        print(f"Type error: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    try:
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
    except TypeError as e:
        print(f"Type error: {e}") 
    except ValueError as e:
        print(f"Value error: {e}") 

    # INTEGRALS 
    try:
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

        plt.loglog(N, err_trap, 'o', label='Trapezoidal integrals')
        plt.loglog(N, err_simpson, 's', label='Simpson integrals')

        plt.loglog(N, model(N, a1, b1), '--', label=f'Fit trapeozoidal integrals: y={a1:.3e} x^{b1:.3f}')

        plt.loglog(N, model(N, a2, b2), '--', label=f'Fit simpson integrals: y={a2:.3e} x^{b2:.3f}')
        plt.title("Error comparison")

        plt.xlabel('N')
        plt.ylabel('errors')
        plt.legend()
        plt.grid(True)
        plt.show()
    except TypeError as e:
        print(f"Type error: {e}") 
    except ValueError as e:
        print(f"Value error: {e}")

    # GAUSSIAN QUADRATURE 
    try:

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
    except TypeError as e:
        print(f"Type error: {e}") 
    except ValueError as e:
        print(f"Value error: {e}")

if __name__ == "__main__":
    main()