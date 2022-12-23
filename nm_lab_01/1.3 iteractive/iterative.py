import numpy as np
import copy
from tabulate import tabulate
from math import sqrt

# Нормы
def matrix_norm(a):
    n = len(a)
    row_sum = 0
    max_norm = 0
    for i in range(n):
        for j in range(n):
           row_sum += abs(a[i][j])
        if row_sum > max_norm:
            max_norm = row_sum
        row_sum = 0
    return max_norm

def distance_norm(x, y):
    norm = 0
    for i in range(len(y)):
            norm += (y[i] - x[i])**2
    norm = sqrt(norm)
    return norm

# Обратная
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

def lup_invert(a):
    c, p = lup(a)
    E = np.eye(len(c))

    inv = []
    for e in E:
        x = lup_solve(c, e, p)
        inv.append(x)
    return np.array(inv).T

# Методы итераций
def resolve_matrices(a, b):
    n = len(a)
    
    alpha = np.zeros((n,n))
    beta = np.zeros(n)

    for i in range(n):
        beta[i] = b[i] / a[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = - a[i][j] / a[i][i]
            else:
                alpha[i][j] = 0

    return alpha, beta

def is_end_of_iterations(a, x_old, x_new, eps):
    a_norm = matrix_norm(a)
    v_norm = distance_norm(x_new, x_old)

    if a_norm >= 1:
        ans = v_norm
    else:
        ans = a_norm / (1 - a_norm) * v_norm
    return ans <= eps

def iterative(a, b, eps=0.001):
    alpha, beta = resolve_matrices(a, b)
    
    x = np.dot(alpha, beta)
    x = x + beta
    it = 0
    while True:
        x_prev = copy.deepcopy(x)
        x = np.dot(alpha, x)
        x = x + beta
        print(f'{it + 1}:', x) # debug
        if is_end_of_iterations(alpha, x_prev, x, eps):
            break
        it += 1
    return x     

def seidel(a, b, eps=0.001):
    n = len(a)
    count = len(b)
    x = np.array([0 for _ in range(count)])
    
    alpha, beta = resolve_matrices(a, b)
    E = np.eye(n)
    B = np.tril(alpha, -1)
    C = alpha - B
    B_inv = lup_invert(E - B)
    beta = np.dot(B_inv, beta)

    it = 0
    while True:
        x_prev = copy.deepcopy(x)
        x = B_inv @ C @ x
        x = x + beta
        print(f'{it + 1}:', x) # debug
        if is_end_of_iterations(alpha, x_prev, x, eps):
            break
        it += 1
    return x    

def main():
    np.set_printoptions(precision=8)
    n = int(input())
    eps = float(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    b = np.array([float(input()) for _ in range(n)])
    print("Matrix A:\n", tabulate(a), '\n')
    print("Matrix B:\n", b, '\n')

    print("Iterative method:")
    x = iterative(a, b, eps)
    print("x =", x)
    print()
    print("Gauss-Seidel method:")
    x = seidel(a, b, eps) 
    print("x =", x)

    print("\nNumpy:")
    print("x =", np.linalg.solve(a, b))


if __name__ == "__main__":
    main()
