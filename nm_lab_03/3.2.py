from progonka import tma
import numpy as np
import matplotlib.pyplot as plt

def function(a, b, c, d, x):
    return a + b * x + c * (x ** 2) + d * (x ** 3)

def count_h(x, i):
    return x[i] - x[i - 1]

def count_a(f):
    return [0] + [f[i] for i in range(len(f) - 1)]

def count_b(f, c, h):
    n = len(f) - 1
    b = [0]

    for i in range(1, n):
        b.append((f[i] - f[i - 1]) / h[i] - 1 / 3 * h[i] * (c[i + 1] + 2 * c[i]))

    b.append((f[n] - f[n - 1]) / h[n] - 2 / 3 * h[n] * c[n])
    return [i for i in b]

def count_c(f, h):
    n = len(f)
    a = [0] + [h[i - 1] for i in range(3, n)]
    b = [2 * (h[i - 1] + h[i]) for i in range(2, n)]
    c = [h[i] for i in range(2, n - 1)] + [0]
    d = [3 * ((f[i] - f[i - 1]) / h[i] - ((f[i - 1] - f[i - 2]) / h[i - 1])) for i in range(2, n)]
    x = tma(a, b, c, d)
    res = [0, 0] + [i for i in x]
    return res

def count_d(h, c):
    n = len(c) - 1
    d = [0]
    for i in range(1, n):
        d.append((c[i + 1] - c[i]) / (3 * h[i]))
    d.append(-c[n] / (3 * h[n]))
    return [i for i in d]

def get_interval(x, x_check):
    for i in range(len(x) - 1):
        if x[i] <= x_check <= x[i + 1]:
            return i

def spline(x, f, x_check):
    n = len(x)
    h = [0]
    for i in range(1, n):
        h.append(count_h(x, i))
    a = count_a(f)
    c = count_c(f, h)
    b = count_b(f, c, h)
    d = count_d(h, c)
    print('h = ', h)
    print('a = ', a)
    print('b = ', b)
    print('c = ', c)
    print('d = ', d, '\n')

    tmp = get_interval(x, x_check)

    ans = function(a[tmp+1], b[tmp+1], c[tmp+1], d[tmp+1], x_check - x[tmp])
    return ans, a, b, c, d

def show_plot(x, f, a, b, c, d):
    X, Y = [], []
    for i in range(len(x) - 1):
        x_i = np.linspace(x[i], x[i + 1], 10, endpoint=True)
        y_i = [function(a[i + 1], b[i + 1], c[i + 1], d[i + 1], j - x[i]) for j in x_i]
        X.append(x_i)
        Y.append(y_i)

    _, ax = plt.subplots()
    ax.scatter(x, f, color='r')
    ax.plot(x, f, color='m')
    for i in range(len(x) - 1):
        ax.plot(X[i], Y[i], color='b')
    plt.show()

def print_spline(a, b, c, d, x):
    print("Spline as object")
    for i in range(len(x)-1):
        print(a[i+1], "+", b[i+1],"*(x -",x[i+1],") + ",c[i+1],"*(x -",x[i+1],")^2 + ",d[i+1],"*(x -",x[i+1],")^3", ", x in [",x[i],',',x[i+1],']\n')

def main():
    x_i = [-3.0, -1.0, 1.0, 3.0, 5.0]
    f_i = [2.8198, 2.3562, 0.7854, 0.32175, 0.1974]
    x_check = -0.5

    print('Coefficients of Spline function:')
    y, a, b, c, d = spline(x_i, f_i, x_check)
    print_spline(a, b, c, d, x_i)
    
    print('Value of function in point X*:')
    print(f'F({x_check}) = {y}')
    show_plot(x_i, f_i, a, b, c, d)

if __name__ == '__main__':
    main()