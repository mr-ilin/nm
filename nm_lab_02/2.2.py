from LU import *
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def x1(x2):
    return np.cos(x2) + 3.0

def x2(x1):
    return np.sin(x1) + 3.0

def f1(x):
    return x[0] - np.cos(x[1]) - 3.0

def f2(x):
    return x[1] - np.sin(x[0]) - 3.0

def phi1(x):
    return 3.0 + np.cos(x[1])

def phi2(x):
    return 3.0 + np.sin(x[0])

def sign_str(a, b):
    return " > " if a > b else " <= "

def derivative(x, f1=False, f2=False, phi1=False, phi2=False, x1=False, x2=False):
    if f1 and x1:
        return 1.0
    elif f1 and x2:
        return np.sin(x[1])
    elif f2 and x1:
        return -np.cos(x[0])
    elif f2 and x2:
        return 1.0

    elif phi1 and x1:
        return 0.0
    elif phi1 and x2:
        return -np.sin(x[1])
    elif phi2 and x1:
        return np.cos(x[0])
    elif phi2 and x2:
        return 0.0


def get_q(x):
    max_phi1 = (abs(derivative(x, phi1=True, x1=True)) + abs(derivative(x, phi1=True, x2=True)))
    max_phi2 = (abs(derivative(x, phi2=True, x1=True)) + abs(derivative(x, phi2=True, x2=True)))
    return max(max_phi1, max_phi2)


def jacobi(x):
    return [[derivative(x, f1=True, x1=True), derivative(x, f1=True, x2=True)],
            [derivative(x, f2=True, x1=True), derivative(x, f2=True, x2=True)]]


def det(mat):
    return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

def delta_x(x):
    # f(x) + J(x) * delta(x) = 0
    A = np.array(jacobi(x))
    b = np.array([-f1(x), -f2(x)])
    pi = LUP(A, len(A))
    L, U = get_LU(A)
    delta = LUP_solve(L, U, pi, b, len(A))
    delta = np.array(delta)
    return delta


def simple_iteration(eps):
    data = [('iterarion', 'x1\nx2', 'phi1\nphi2', 'eps')]

    count_iterations = 0
    x_last = [2.5, 3.5]
    q = get_q(x_last)

    data.append(create_data_item(count_iterations, x_last, '-'))

    while True:
        count_iterations += 1
        x = [phi1(x_last), phi2(x_last)]
        finish_it = max([abs(i - j) for i, j in zip(x, x_last)]) * q / (1 - q)

        data.append(create_data_item(count_iterations, x, finish_it))

        if finish_it <= eps:
            break
        x_last = x

    return data


def newton(eps):
    data = [('iterarion', 'x1\nx2', 'phi1\nphi2', 'eps')]

    count_iterations = 0
    x_last = [2.1, 3.9]

    data.append(create_data_item(count_iterations, x_last, '-'))

    while True:
        count_iterations += 1
        delta = delta_x(x_last)
        x = [x_last[0] + delta[0],
             x_last[1] + delta[1]]
        finish_it = max([abs(i - j) for i, j in zip(x, x_last)])

        data.append(create_data_item(count_iterations, x, finish_it))

        if finish_it <= eps:
            break

        x_last = x

    return data

def plot(x1, x2, step = 0.1):
        X = np.arange(0, 10, step)
        Y1 = [x1(i) for i in X]
        Y2 = [x2(i) for i in X]

        _, axis = plt.subplots()
        axis.set_title(f'Scatter x1 from x2')
        axis.plot(X, Y1, label='func1')
        axis.plot(Y2, X, label='func2')
        axis.legend(loc='upper right')
        axis.grid()

        plt.show()

def create_data_item(iteration, x, eps):
    return (
        iteration,
        '{}\n{}'.format(x[0], x[1]),
        '{}\n{}'.format(phi1(x), phi2(x)),
        eps
    )

def print_data(name, data):
    print(name)
    print(tabulate(data, headers='firstrow', tablefmt="grid", stralign='left'))
    print("\n")

def main():
    eps = 0.01

    iterations_data = simple_iteration(eps)
    print_data('Simple iterations', iterations_data)

    newton_data = newton(eps)
    print_data('Newton', newton_data)

    plot(x1,x2)

if __name__ == '__main__':
    main()
