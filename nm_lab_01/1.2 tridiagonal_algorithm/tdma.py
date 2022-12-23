import numpy as np
from tabulate import tabulate


def tridiagonal_matrix_algorithm(a, b, c, d):
    n = len(d)

    p = [0] * n
    q = [0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        y = b[i] + a[i - 1] * p[i - 1]
        if i != n-1:
            p[i] = -c[i] / y
        q[i] = (d[i] - a[i - 1] * q[i - 1]) / y

    x = [0] * n
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x

def get_diagonals(m):
    n = len(m)
    a, b, c = [], [], [] 
    
    for i in range(n):
        if i != 0:
            a.append(m[i][i - 1])
        if i != n - 1:
            c.append(m[i][i + 1])
        b.append(m[i][i])
    return a, b, c


def main():
    np.set_printoptions(precision=6)
    n = int(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    b = np.array([float(input()) for _ in range(n)])

    x, y, z = get_diagonals(a)

    print("Matrix A:\n", tabulate(a), '\n')
    print("Matrix B:")
    print(b, '\n')

    print("TDMA:")
    x = tridiagonal_matrix_algorithm(x, y, z, b)
    print('x =', x, '\n')

    print("Numpy:")
    print('x =', np.linalg.solve(a, b))


if __name__ == "__main__":
    main()
