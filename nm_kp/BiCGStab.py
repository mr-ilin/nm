from tokenize import Double
import numpy as np
from random import randint
from scipy.sparse import rand
from numpy.linalg import norm
from scipy.sparse import csc_matrix
import time

class BiCGStab:
    def __init__(self, matrix, b, x0=None, eps=1e-5):
        self.matrix = matrix
        self.b = b
        self.eps = eps
        self.shape = matrix.shape[0]
        self.x0 = np.array([0] * self.shape) if x0 is None else x0
        self.k = 0

    def solve(self):
        r0 = self.b - self.matrix @ self.x0 # невязка
        x0 = self.x0
        r2 = r0
        rho0 = 1
        alpha0 = 1
        omega0 = 1
        v0 = np.array([0] * self.shape)
        p0 = np.array([0] * self.shape)
        while True:
            rho = r2 @ r0
            beta = np.float64((rho * alpha0) / (rho0 * omega0))
            p = r0 + beta * (p0 - omega0 * v0)
            v = self.matrix @ p
            alpha = rho / (r2 @ v)
            s = r0 - alpha * v
            t = self.matrix @ s
            omega = (t @ s) / (t @ t)
            x = x0 + omega * s + alpha * p
            r = s - omega * t

            self.k += 1
            if norm(r) < self.eps:
                break
            r0 = r
            rho0 = rho
            alpha0 = alpha
            omega0 = omega
            v0 = v
            p0 = p
            x0 = x
        return x

    def print_solution(self):
        start_timeBiCGM = time.time()
        x = self.solve()
        print("BiCGStab time: %s seconds" % (time.time() - start_timeBiCGM))
        start_timeNumPy = time.time()
        x2 = np.linalg.solve(self.matrix.toarray(), self.b)
        print("NumPy time: %s seconds\n" % (time.time() - start_timeNumPy))
        print('My solve:')
        print(f'{x.round(5)}\n')
        print(f'Eps = {self.eps}')
        print(f'Line = {self.shape}')
        print(f'Count of iterations = {self.k}')
        print(f'Mean = {np.mean(x)}') # среднее значение
        print('\nNumPy solve:')
        print(f'{x2.round(5)}\n')
        print(f'Mean = {np.mean(x2)}')

def main():
    shape = int(input())
    if shape < 3:
        exit()

    matrix = rand(shape, shape, density = 0.4, random_state = randint(112, 154))

    for i in matrix.toarray().round(3):
        for j in i:
            print(f'{j} ', end=" ")
        print('\n')
    print('\n')
    b = np.random.randint(5, 53, shape)
    for i in b:
        print(f'{i} ', end=" ")
    print('\n')

    matrix = csc_matrix(matrix)

    solver = BiCGStab(matrix, b, eps=1e-5)
    solver.print_solution()

if __name__ == '__main__':
    main()
