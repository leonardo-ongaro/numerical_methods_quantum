def factorial(n):
    fact = 1    

    # the factorial is the product of all the numbers smaller than n
    for i in range(1, n+1):     
        fact *= i       

    return fact 

def fibonacci(n):
    fib = []

    a, b = 0, 1     # first two numbers of the Fibonacci series
    
    # the first two numbers are manually added to the succession
    if n >= 1:
        fib.append(a)
    if n >= 2:
        fib.append(b)

    # compute the Fibonacci sequence --> f(n) = f(n-1) + f(n-2)
    for i in range(2, n):  
        a, b = b, a + b
        fib.append(b)

    return fib 

def main():
    num = 0
    while True:
        inp = input("Enter a non-negative integer (default 10): ")

        if inp == "": # default number
            num = 10
        else:
            num = int(inp)

        if num >= 0: # check if the number is non-negative
            break

    fact = factorial(num)
    fib = fibonacci(num)

    print(f"The factorial of {num} is {fact}")
    print(f"The Fibonacci sequence up to {num} is {fib}")

if __name__ == "__main__":
    main()
