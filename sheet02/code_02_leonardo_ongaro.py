import numpy as np
from decimal import Decimal
import math
import numbers

def integer_overflow():
    a16 = np.int16(32767) # 32767 is the largest int with 16 bits
    a32 = np.int32(32767) # same value but stored as a 32 bits integer

    print(f"If we add 1 to 32767 stored as a 16 bits integer we obtain {a16 + np.int16(1)}") # Runtime warning: integer overflow
    print(f"If we add 1 to 32767 stored as a 32 bits integer we obtain {a32 + np.int32(2)}")
    print()

def float_precision():
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

def machine_epsilon(dtype = float):

    if not np.issubdtype(dtype, np.floating): # raise an error if the input is not a floating point 
        raise TypeError("The data type must be a floating point") # this error can be catched in the main
    
    # declaring epsilon and one to be of type dtype
    epsilon = dtype(1)
    one = dtype(1)

    while one + epsilon != one:
        epsilon /= dtype(2)

    epsilon *= dtype(2)

    return epsilon

def float_comparison():
    print("Is 0.1 + 0.2 == 0.3?")
    print(0.1 + 0.2 == 0.3)
    
    print("Is 0.1 + 0.2 close to 0.3?")
    print(math.isclose(0.1 + 0.2, 0.3))

def f(x): 
    if not isinstance(x, numbers.Number) and not isinstance(x, np.number):
        raise TypeError("The type of x must be a number")
    return (1-np.cos(x))/x**2



if __name__ == "__main__":
    # DATA TYPES AND PRECISION

    integer_overflow()
    float_precision()

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
    float_comparison()
    # catastophic cancellation
    print("Evaluation of (1-cos(x))/x**2 for: ")
    try:
        print(f"x = 10 ** -4: {f(10**-4)}")
        print(f"x = 10 ** -8: {f(10**-8)}")
        print(f"x = 10 ** -12: {f(10**-12)}")
    except TypeError as e:
        print(f"Error: {e}")

