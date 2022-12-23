import numpy as np
import copy
class MatrixException(Exception):
    pass

def outp(A, n):
    for i in range(0, n):
        for j in range(0, n):
            print("\t", A[i][j], " ", end="")
        print("\n")


def LUP(A,n):
    pi = [i for i in range(n)]

    for col in range(n):
        pval = 0
        pivot = -1
        for row in range(col , n):
            if abs(A[row][col]) > pval:
                pval = abs(A[row][col])
                pivot = row
            if pval == 0:
                raise MatrixException("Matrix is degenerate")

        pi[col], pi[pivot] = pi[pivot] , pi[col]

        for row in range(n):
            A[col][row], A[pivot][row] = A[pivot][row], A[col][row]

        for row in range(col+1, n):
            A[row][col] = A[row][col] / A[col][col]
            for j in range(col + 1,n):
                A[row][j] = A[row][j] - A[row][col] * A[col][j]
    return pi



def get_LU(A):
    n = len(A)
    L = [[0] * n for i in range(0, n)]
    U = [[0] * n for i in range(0, n)]

    for i in range(n):
        L[i][i] = 1
        for j in range(n):
            if j < i:
                L[i][j] = A[i][j]
            else:
                U[i][j] = A[i][j]
    return L, U

def LUP_solve(L, U, pi,b,n):
    x = [i for i in range(n)]
    z = [i for i in range(n)]

    for row in range(n):
        sum = 0
        for col in range(row):
            sum += L[row][col] * z[col]
        z[row] = b[pi[row]] - sum
    for row in range(n-1,-1,-1):
        sum = 0
        for col in range(row+1,n):
            sum += U[row][col] * x[col]
        x[row] = (z[row] - sum) / U[row][row]

    return x


def determinant(U, n): # LU
    D = 1
    for i in range(n):
        D *= U[i][i]
    return D

def LU_inv(L,U,pi,n):
    E = np.eye(n)
    x = []
    for row in E:
        x.append(LUP_solve(L,U,pi,row,n))
    inv = np.array(x)
    inv = inv.T
    return inv


if __name__ == '__main__':

    n = int(input())
    A = []
    b = []
    for i in range(n):
        A.append(list(map(float, input().split())))
    b.append(list(map(float, input().split())))
    b = sum(b, [])
    print("Start A:")
    outp(A, n)
    A1 = copy.deepcopy(A)
    pi = LUP(A, n)
    print("A after LUP")
    outp(A,n)
    print("pivotes:", pi)
    L, U = get_LU(A)
    print("L:")
    outp(L, n)
    print("U:")
    outp(U, n)
    print("Checking A:\n")
    R = np.array(L) @ np.array(U)
    outp(R, n)
    print("Determinant:")
    d = determinant(U,n)
    print(d)
    print("Solving:")
    print(LUP_solve(L, U, pi, b, n))
    print("Inverse Matrix:")
    inv = LU_inv(L, U, pi, n)
    outp(inv,n)
    print("Checking inverse matrix ")
    res = np.array(inv) @ np.array(A1)
    outp(res,n)

