import numpy as np
from tabulate import tabulate

def lup(a):
    n = len(a)
    c = a.copy()
    p = [i for i in range(n + 1)]
    p[n] = 0

    for i in range(n):
        pivot_value = 0
        pivot = -1
        for row in range(i, n):
            if np.abs(c[row][i]) > pivot_value:
                pivot_value = np.abs(c[row][i])
                pivot = row

        if pivot_value != 0:
            p[i], p[pivot] = p[pivot], p[i]
            c[[pivot, i]] = c[[i, pivot]]
            p[n] += 1
            
            for j in range(i + 1, n):
                c[j][i] /= c[i][i]
                for k in range(i + 1, n):
                    c[j][k] -= c[j][i] * c[i][k]
    return c, p

def lup_solve(c, b, p):
    n = len(c)
    x = [None] * n
    for i in range(n):
        x[i] = b[p[i]]
        for k in range(i):
            x[i] -= c[i][k] * x[k]
    
    for i in range(n - 1, -1, -1):
        for k in range(i + 1, n):
            x[i] -= c[i][k] * x[k]
        x[i] /= c[i][i]
    return x

def lup_invert(c, p):
    E = np.eye(len(c))

    inv = []
    for e in E:
        x = lup_solve(c, e, p)
        inv.append(x)
    return np.array(inv).T

def lup_determinant(c, p):
    n = len(c)

    det = c[0][0]
    for i in range(1, n):
        det *= c[i][i]

    if all(a + 1 == b for a, b in zip(p[:n - 1], p[1:n - 1])):
        return det
    return det if p[n] % 2 == 0 else -det

def main():
    np.set_printoptions(precision=2)
    n = int(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    b = [float(input()) for _ in range(n)]

    print("Matrix A:", tabulate(a), sep='\n')
    print("\nMatrix B:")
    print(b)

    c, p = lup(a)

    print("\nL")
    l = np.tril(c)
    for i in range(len(c)):
        for j in range(len(c)):
            if i == j:
                l[i][j] = 1
    print(tabulate(l))

    print("U")
    u = np.triu(c)
    print(tabulate(u))
    
    print("P")
    print(p)
    print("\n")

    print("L * U")
    print(tabulate(np.dot(l, u)))

    print("\nLUP decomposition:")
    print("x =", lup_solve(c, b, p), '\n')
    print("Det =", lup_determinant(c, p), '\n')
    print("Inverse\n", tabulate(lup_invert(c, p)))

    print("\nNumpy:")
    print("x =", np.linalg.solve(a, b), '\n')
    print("Det =", np.linalg.det(a), '\n')
    print("Inverse\n", tabulate(np.linalg.inv(a)))



if __name__ == "__main__":
    main()
