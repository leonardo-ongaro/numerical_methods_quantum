def factorial(n):
    fact = 1    

    for i in range(1, n+1):     # loop over 1 to n included
        fact *= i       # the factorial is the product of all the numbers below n

    return fact 

def fibonacci(n):
    fib = []

    a, b = 0, 1     # first two numbers of the Fibonacci series
    
    # the first two numbers are manually added to the succession
    if n >= 1:
        fib.append(a)
    if n >= 2:
        fib.append(b)

    for i in range(2, n):  # implementation of f(n) = f(n-1)+f(n-2)
        a, b = b, a + b
        fib.append(b)

    return fib 

def main():
    print("Factorial of 10 and first 10 elements of the Fibonacci suquence")

    n = 10

    fact = factorial(n)
    fib = fibonacci(n)

    print(f"The factorial of {n} is {fact}")
    print(f"The Fibonacci sequence up to {n} is {fib}")

if __name__ == "__main__":
    main()
