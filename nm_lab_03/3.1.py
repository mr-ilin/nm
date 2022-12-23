import numpy as np

def function(x):
    return np.arctan(x)

def denominator(i, x):
    n = len(x)
    r = [1.0]
    for j in range(n):
        if j != i:
            r.append(x[j])
            r[0] /= x[i] - x[j]
    return r

def lagrange(x):
    n = len(x)
    y = [function(val) for val in x]
    L = []
    for i in range(n):
        l = denominator(i, x)
        l[0] *= y[i]
        L.append(l)

    return L

def newton(x):
    n = len(x)
    y = [[function(val) for val in x]]
    k = 1
    N = [[y[0][0]]]
    for i in range(n - 1):
        y.append([])
        for j in range(len(y[i]) - 1):
            y[i + 1].append((y[i][j] - y[i][j + 1]) / (x[j] - x[j + k]))
        N.append([y[i+1][0]] + [x[k] for k in range(i + 1)])
        k += 1
    return N

def print_polynom(polynom):
    for each in polynom:
        if each[0] < 0:
            print(f'- {abs(each[0]):5.3f}', end=' ')
        else:
            print(f'+ {abs(each[0]):5.3f}', end=' ')
        for i in range(1, len(each)):
            if each[i] > 0:
                print(f'(x - {each[i]:3.1f})', end=' ')
            else:
                print(f'(x + {abs(each[i]):3.1f})', end=' ')
    print('\n', end='')

def sub(P):
    r = 1
    for i in P:
        r *= i
    return r

# это че бл
def count_result(x, P):
    function = 0
    for coeff in P:
        function += coeff[0] * sub(x - coeff[i] for i in range(1, len(coeff)))
    return function

def main():
    x_1 = [-3.0, -1.0, 1.0, 3.0]
    x_2 = [-3.0, 0.0, 1.0, 3.0]
    x_0 = -0.5

    print('Lagrange')
    L = lagrange(x_1)
    print("L(x) =", end=' ')
    print_polynom(L)

    Lx_0 = count_result(x_0, L)
    print(f'L({x_0}) = {Lx_0:6.3f}\n')

    print('Newton')
    N = newton(x_2)
    print("N(x) =", end=' ')
    print_polynom(N)

    Nx_0 = count_result(x_0, N)
    print(f'N({x_0}) = {Nx_0:6.3f}\n')

    print('Function')
    Fx = function(x_0)
    print(f'F({x_0}) = {Fx:6.3f}\n')
    print('Error')
    print(f'|F(x) - L(x)| = {abs(Fx - Lx_0):7.4}')
    print(f'|F(x) - N(x)| = {abs(Fx - Nx_0):7.4}')

if __name__ == '__main__':
    main()
