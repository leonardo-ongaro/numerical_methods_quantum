import numpy as np
from decimal import Decimal
import math
import numbers


def machine_epsilon(dtype = float):

    """Compute the machine epsilon for a given floating-point data type.

    Parameters
    ----------
    dtype : type, optional
        Floating-point data type used for the computation. Defaults to float.

    Returns
    -------
    float
        The machine epsilon, defined as the smallest positive floating-point
        value for which adding it to 1 produces a result different from 1.

    Raises
    ------
    TypeError
        If the specified data type is not a floating-point type.
    """



    if not np.issubdtype(dtype, np.floating): # raise an error if the input is not a floating point 
        raise TypeError("The data type must be a floating point") # this error can be catched in the main
    
    epsilon = dtype(1)
    one = dtype(1)

    while one + epsilon != one:
        epsilon /= dtype(2)

    epsilon *= dtype(2)

    return epsilon


def f(x): 
    """
    Evaluate the function (1 - cos(x)) / x**2.

    Parameters
    ----------
        x : int, float, or numpy number
        Numeric value at which the function is evaluated.

    Returns
    -------
        float or numpy number
        The result of evaluating (1 - cos(x)) / x**2.

    Raises
    ------
        TypeError
        If x is not a numeric value.
    """

    if not isinstance(x, numbers.Number) and not isinstance(x, np.number):
        raise TypeError("The type of x must be a number")
    return (1-np.cos(x))/x**2

def f_better(x): 
    """
    Evaluate the function 2 * sin(x / 2) ** 2.

    Parameters
    ----------
        x : int, float, or numpy number
        Numeric value at which the function is evaluated.

    Returns
    -------
        float or numpy number
        The result of evaluating 2 * sin(x / 2) ** 2.

    Raises
    ------
        TypeError
        If x is not a numeric value.
    """
        
    if not isinstance(x, numbers.Number) and not isinstance(x, np.number):
        raise TypeError("The type of x must be a number")
    return 2*np.sin(x/2)**2

def factorial(n):
    """
    Calculate the factorial of a non-negative integer.

    Args:
        n (int): The number whose factorial is to be calculated.

    Returns:
        int: The factorial of n.

    Raises:
        TypeError: If n is not an integer.
        ValueError: If n is a negative integer.
    """
    if not isinstance(n, numbers.Integral):
        raise TypeError("n must be an integer.")

    if n < 0:
        raise ValueError("n cannot be negative.")

    fact = 1

    # The factorial is the product of all integers from 1 to n
    for i in range(1, n + 1):
        fact *= i

    return fact

assert factorial(5) == 120

def fibonacci(n):
    """
    Generate the first n numbers of the Fibonacci sequence.

    Args:
        n (int): The number of Fibonacci numbers to generate.

    Returns:
        list: A list containing the first n Fibonacci numbers.

    Raises:
        TypeError: If n is not an integer.
        ValueError: If n is a negative integer.
    """
    if not isinstance(n, numbers.Integral):
        raise TypeError("n must be an integer.")

    if n < 0:
        raise ValueError("n cannot be negative.")

    fib = []

    a, b = 0, 1  # First two numbers of the Fibonacci sequence

    # Add the first two numbers to the sequence
    if n >= 1:
        fib.append(a)
    if n >= 2:
        fib.append(b)

    # Compute the remaining Fibonacci numbers:
    # f(n) = f(n-1) + f(n-2)
    for i in range(2, n):
        a, b = b, a + b
        fib.append(b)

    return fib

assert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]

def checkpoint (*values, msg = "", debug = True):
    """ Print a labelled debug message, silenced when debug=False"""
    if debug:
        label = f"[checkpoint] {msg}" if msg else "[checkpoint]"
    print(label, *values)

if __name__ == "__main__":
    # DATA TYPES AND PRECISION
    a16 = np.int16(32767) # 32767 is the largest int with 16 bits
    a32 = np.int32(32767) # same value but stored as a 32 bits integer

    print(f"If we add 1 to 32767 stored as a 16 bits integer we obtain {a16 + np.int16(1)}") # Runtime warning: integer overflow
    print(f"If we add 1 to 32767 stored as a 32 bits integer we obtain {a32 + np.int32(2)}")
    print()

    f32_1 = np.float32(np.pi * 10 ** 32) # pi * 10 ** 32 as a float32
    f64_1 = np.float64(np.pi * 10 ** 32) # pi * 10 ** 32 as a float64
    f32_2 = np.float32(np.sqrt(2) * 10 ** 21) # sqrt(2) * 10 ** 21 as a float32
    f64_2 = np.float64(np.sqrt(2) * 10 ** 21) # sqrt(2) * 10 ** 21 as a float64
    
    d1 = Decimal("3.141592653589793238462643383279502884197") * 10 ** 32 # pi * 10 ** 32 as a Decimal 
    d2 = Decimal("2").sqrt() * 10 ** 21 # sqrt(2) * 10 ** 21 as a Decimal
    
    print("pi * 10 ** 32: ")
    print(f"as a float32 {f32_1}")
    print(f"as a float64 {f64_1}")
    print(f"as a Decimal {d1}")
    print()
    print("sqrt(2) * 10 ** 21: ")
    print(f"as a float32 {f32_2}")
    print(f"as a float64 {f64_2}")
    print(f"as a Decimal {d2}")
    print()

    try:
        print("Machine epsilon for a float32: ")
        print(f"with function machine_epsilon: {machine_epsilon(np.float32)}")
        print(f"with np.finfo: {np.finfo(np.float32).eps}")
        print()
        print("Machine epsilon for a float64: ")
        print(f"with function machine_epsilon: {machine_epsilon(np.float64)}")
        print(f"with np.finfo: {np.finfo(np.float64).eps}")
        print()
    except TypeError as e:
        print(f"Error: {e}")

    # NUMERICAL STABILITY AND COMMON PITFALLS
    # float comparison
    print("Is 0.1 + 0.2 == 0.3?")
    print(0.1 + 0.2 == 0.3)
    
    print("Is 0.1 + 0.2 close to 0.3?")
    print(math.isclose(0.1 + 0.2, 0.3))
    # catastophic cancellation
    print("Evaluation of (1-cos(x))/x**2 for: ")
    try:
        print(f"x = 10 ** -4: {f(10**-4)}")
        print(f"x = 10 ** -8: {f(10**-8)}")
        print(f"x = 10 ** -12: {f(10**-12)}")
    except TypeError as e:
        print(f"Error: {e}")

    print("Evaluation of 2sin(x/2)**2 for: ")
    try:
        print(f"x = 10 ** -4: {f_better(10**-4)}")
        print(f"x = 10 ** -8: {f_better(10**-8)}")
        print(f"x = 10 ** -12: {f_better(10**-12)}")
    except TypeError as e:
        print(f"Error: {e}")

    # SUMMATION ORDER MATTERS
    l = [10**16] + [1.0] * 10**6
    sum = 0
    for i in range(len(l)):
        sum += l[i]
        if i % 10**5 == 0:
            checkpoint(sum, msg = "", debug = True) 
    print(f"The sum in the given order is {sum}")
    sum = 0
    for i in range(len(l)-1, -1, -1):
        sum += l[i]
        if i % 10**5 == 0:
            checkpoint(sum, msg = "", debug = True) 
    print(f"The sum with the small numbers first and the largest last is {sum}")
    print(f"The sum using math.fsum is {math.fsum(l)}")

    # SILENT VS LOUD FAILURE 
    print("This will raise an error, to see this uncomment the line of code below")
    #np.seterr(over='raise')
    print(f"If we add 1 to 32767 stored as a 16 bits integer we obtain {a16 + np.int16(1)}")

    # FACTORIAL 
    num = 5
    try:
        result = factorial(num)
        print(f"Factorial of {num}:", result)

    except TypeError as e:
        print("Type error:", e)
    except ValueError as e:
        print("Value error:", e)

    # FIBONACCI
    num = 7
    try:
        result = fibonacci(7)
        print(f"Fibonacci sequence with {num} values:", result)
    except TypeError as e:
        print("Type error:", e)
    except ValueError as e:
        print("Value error:", e)