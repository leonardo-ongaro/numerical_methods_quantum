"""
Module for calculating the factorial of a non-negative integer.

This module provides:
- checkpoint(): Prints labelled debug messages when debugging is enabled.
- factorial(): Calculates the factorial of a non-negative integer
  and raises exceptions for invalid inputs.

Example:
    >>> factorial(5)
    120

The module also includes a simple example of how to use the factorial()
function when executed as a standalone script.
"""

import numbers

def checkpoint (*values, msg = "", debug = True):
    """ Print a labelled debug message, silenced when debug=False"""
    if debug:
        label = f"[checkpoint] {msg}" if msg else "[checkpoint]"
    print(label, *values)

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
        if n >= 4 and i % (n // 4) == 0:
            checkpoint(fact, msg="Intermediate value of the factorial", debug=True)
        fact *= i

    return fact

assert factorial(5) == 120

if __name__ == "__main__":
    num = 5
    try:
        result = factorial(num)
        print(f"Factorial of {num}:", result)
    
    except TypeError as e:
        print("Type error:", e)
    except ValueError as e:
        print("Value error:", e)