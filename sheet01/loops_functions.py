import numpy as np

def factorial(n):
    fact = 1

    for i in range(1, n+1):
        fact *= i

    return fact 

def fibonacci(n):
    fib = np.array([])

    a = 0
    b = 1

    if n >= 1:
        fib = np.append(fib, a)
    if n >= 2:
        fib = np.append(fib, b)

    for i in range(2, n):
        temp = b
        b = a + b
        a = temp

        fib = np.append(fib, b)

    return fib 

def main():
    n = int(input("Enter a non negative integer: "))

    fact = factorial(n)
    fib = fibonacci(n)

    print(f"The factorial of {n} is {fact}")
    print(f"The Fibonacci sequence up to {n} is {fib}")

if __name__ == "__main__":
    main()
