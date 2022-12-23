from LU import *
import matplotlib.pyplot as plt

def function(x, coeffs):
    return sum([c * (x ** i) for i, c in enumerate(coeffs)])

def se(f, y):
    return sum([(f_i - y_i) ** 2 for f_i, y_i in zip(f, y)])

def mls(n, x, y):
    A = [[] for _ in range(n + 1)]
    size = len(A)
    for i in range(n + 1):
        for j in range(n + 1):
            A[i].append(sum([x_j ** (i + j) for x_j in x]))

    b = [0 for _ in range(n + 1)]
    for i in range(n + 1):
        b[i] = sum([y_j * (x_j ** i) for x_j, y_j in zip(x, y)])

    P = LUP(A, size)
    L, U = get_LU(A)
    a = LUP_solve(L, U, P, b, size)
    return [i for i in a]

def get_polynom_str(coeffs):
    n = len(coeffs)
    f = f'F{n - 1}(x) = '
    for i in range(n):
        f += f'{coeffs[i]}x^{i} + '
    f = f[:-2]
    return f

def plot(l, r, xs, ys, coefs):
    _, ax = plt.subplots(1, 1)
    plot_x = np.linspace(l, r, 100)
    plot_y1 = coefs[0][0] + coefs[0][1]*plot_x
    plot_y2 = coefs[1][0] + coefs[1][1]*plot_x + coefs[1][2]*plot_x*plot_x
    ax.plot(plot_x, plot_y1, "-r")
    ax.plot(plot_x, plot_y2, "-b")
    ax.scatter(xs, ys)
    plt.show()

def main():
    x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y = [-1.8415, 0.0, 1.5403, 1.5839, 2.010, 3.3464]
    F = []
    error = []
    coeffs = []

    for power in [1, 2]:
        print(f'Power = {power}')
        coeffs.append(mls(power, x, y))
        print(get_polynom_str(coeffs[power - 1]))
        F.append([function(i, coeffs[power - 1]) for i in x])
        error.append(se(F[power - 1], y))

    print()
    k = 1
    for i in error:
        print(f'Error of F{k} = {i}')
        k += 1 

    plot(x[0], x[-1], x, y, coeffs)

if __name__ == '__main__':
    main()