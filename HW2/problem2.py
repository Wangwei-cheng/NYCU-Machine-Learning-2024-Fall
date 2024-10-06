import sys
import math

def Factorial(n):
    F = 1
    for i in range(1, n+1):
        F *= i

    return F


def C(a, b):
    return Factorial(a) / (Factorial(b) * Factorial(a-b))

def Binomial(N, m):
    p = m / N
    return C(N, m) * p**m * (1-p)**(N-m)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('no argument')
        sys.exit()

    filepath = sys.argv[1]
    a = int(sys.argv[2])
    b = int(sys.argv[3])

    with open(filepath, 'r') as file:
        data = file.read()

    data = data.split('\n')
    data.remove('')

    for line, line_data in enumerate(data):
        N = len(line_data)
        m = line_data.count("1")

        print(f"case {line+1}: {line_data}")
        print(f"Likelihood: {Binomial(N, m)}")
        print(f"Beta prior:     a = {a}, b = {b}")
        a += m
        b += N-m
        print(f"Beta posterior: a = {a}, b = {b}")
        print()
    
