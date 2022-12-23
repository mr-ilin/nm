import numpy as np
from tabulate import tabulate


def create_u_matrix(a, i, j):
    n = a.shape[0]
    ang = 0.5 * np.arctan(2 * a[i][j] / (a[i][i] - a[j][j]))    
    u = np.eye(n) 
    u[i][j] = -np.sin(ang)
    u[j][i] = np.sin(ang)
    u[j][j] = np.cos(ang)
    u[i][i] = u[j][j]
    return u
    

def find_max(a):
    n = a.shape[0]
    i = 0
    j = 1
    max_ = a[0][1]
    for row in range(n):
        for col in range(row, n):
            if row != col and abs(a[row][col]) > max_:
                max_ = abs(a[row][col])
                i = row
                j = col
    return i, j

def count_error(a):
    n = a.shape[0]
    t = 0
    for row in range(n):
        for col in range(row, n):
            if row != col:
                t += a[row][col] ** 2
    return np.sqrt(t)


def jacobi(a, eps=0.001):
    n = a.shape[0]
    resulting_u = np.eye(n)
    while True:
        i, j = find_max(a)
        u = create_u_matrix(a, i, j)
        resulting_u = resulting_u.dot(u)
        a = (u.T.dot(a)).dot(u)

        if count_error(a) < eps:
            break

    eigen_vectors = np.eye(n)
    eigen_vectors = eigen_vectors.dot(resulting_u)

    eigen_values = []
    for i in range(n):
        eigen_values.append(a[i][i])

    return eigen_vectors, eigen_values


def main():
    n = int(input())
    eps = float(input())
    a = np.array([list(map(float, input().split())) for _ in range(n)])
    print("Matrix A:\n", tabulate(a), '\n')

    vectors, values = jacobi(a, eps)
    print("Jacobi eigen vectors:")
    print(tabulate(vectors))
    print()
    print("Jacobi eigen values:\n", values, sep="")

    print(">>>> \n")
    for (value, vector) in zip(values, vectors.T):
        print("Checking: ", vector, " and ", value)
        print("> ", a.dot(vector), " = ", vector.dot(value), "\n")
    print(">>>> \n")

    values, vectors = np.linalg.eig(a)
    print("\nNumpy eigen vectors:")
    print(tabulate(vectors))
    print()
    print("Numpy eigen values:\n", values, sep="")

if  __name__ == "__main__":
    main()
