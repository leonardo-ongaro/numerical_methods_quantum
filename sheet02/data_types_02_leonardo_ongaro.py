import numpy as np

def epsilon():
    f32 = np.float32(1.0)
    f64 = np.float64(1.0)

    while (np.float32(1.0) + f32 != np.float32(1.0)):
        f32 = f32 / 2

    while (np.float64(1.0) + f64 != np.float64(1.0)):
        f64 = f64 / 2

    return f32 * 2, f64 * 2

def f(x):
    return (1-np.cos(x))/(x**2)

def f_better(x):
    return 2*np.sin(x/2)**2 / x**2

def main():
    # INTEGERS
    print("--- INTEGER OVERFLOW ---")
    a16 = np.int16(32767)
    a32 = np.int32(32767)

    print("Integer with 16 bits: ", a16 + np.int16(1))
    print("Integer with 32 bits: ", a32 + np.int32(1))

    # FLOATS
    print("--- FLOATS ---")
    f32_1 = np.float32(np.pi * 10 ** 32)
    f64_1 = np.float64(np.pi * 10 ** 32)

    print("pi * 10 ^ 32 with 32 bits: ", f32_1)
    print("pi * 10 ^ 32 with 64 bits: ", f64_1)

    f32_2 = np.float32(np.sqrt(2) * 10 ** 21)
    f64_2 = np.float64(np.sqrt(2) * 10 ** 21)

    print("sqrt(2) * 10 ^ 21 with 32 bits: ", f32_2)
    print("sqrt(2) * 10 ^ 21 with 64 bits: ", f64_2)

    # MACHINE EPSILON
    print("--- MACHINE EPSILON ---")
    e32, e64 = epsilon()

    print("Value of epsilon for single precision float: ", e32, ", ", np.finfo(np.float32).eps)
    print("Value of epsilon for double precision float: ", e64, ", ", np.finfo(np.float64).eps)

    # FLOAT COMPARISON 
    print("--- FLOAT COMPARISON ---")
    print("0.1 + 0.2 == 0.3? ", 0.1 + 0.2 == 0.3)

    # CATASTROPHIC CANCELLATION

    print("---CATASTROPHIC CANCELLATION---")
    print("Limit of f(x) = (1-cos(x))/x^2 for x -> 0")
    print("Analytical calculation ", 0.5)

    print("Numerical calculation with x = 10^(-4): ", f(10**(-4)))
    print("Numerical calculation with x = 10^(-8): ", f(10**(-8)))
    print("Numerical calculation with x = 10^(-12): ", f(10**(-12)))

    print("Results with the identity 1 - cos(x) = 2sin^2(x/2)")
    print("Numerical calculation with x = 10^(-4): ", f_better(10**(-4)))
    print("Numerical calculation with x = 10^(-8): ", f_better(10**(-8)))
    print("Numerical calculation with x = 10^(-12): ", f_better(10**(-12)))

    print("--- SUMMATION ORDER MATTERS ---")


if __name__ == "__main__":
    main()
