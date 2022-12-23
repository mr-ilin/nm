import math
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def f1(x):
    return np.log(x+2)
def f2(x):
    return x**4 - 0.5

def f(x):
    return np.log(x+2) - x**4 + 0.5

def df(x):
    return 1/ (x+2) - 3*x**3

def ddf(x):
    return (-1)/((x+2)**2) - 6*x**2

def phi(x):
    return (np.log(x + 2) + 0.5) ** 0.25

def dphi(x):
    return 0.25 / (math.sqrt(np.log(x+2)+0.5)*(x+2))

 
def count_q(dphi, a, b, eps): 
    count = (b - a) / eps
    x = np.linspace(a, b, int(count)) 
    y = [ abs(dphi(i)) for i in x ] 
    q = np.max(y) 
    return q

def simple_iterations(phi, dphi, a, b, eps=0.001):
    data = [('x', 'iterarion', 'epsilon')]
    q = count_q(dphi, a, b, eps)
    x = (a + b) / 2
    count_iterations = 0

    while True:
        count_iterations += 1
        x_cur = phi(x)
        eps_cur = q * abs(x_cur - x) / (1 - q)
        data.append((x_cur, count_iterations, eps_cur))
        if eps_cur <= eps:
            break
        x = x_cur

    return data


def newton(f, df, x0, eps=0.001):
    data = [('x', 'iterarion', 'epsilon')]
    x = x0
    count_iterations = 0
    while True:
        count_iterations += 1
        x_cur = x - f(x) / df(x)
        eps_cur = abs(x_cur - x)
        data.append((x_cur, count_iterations, eps_cur))
        if eps_cur <= eps:
            break
        x = x_cur

    return data


def plot(f, df, x, step = 0.02):
    X = np.arange(x[0], x[1], step)
    Y = [f(i) for i in X]
    dY = [df(i) for i in X]

    _, axis = plt.subplots()
    axis.set_title(f'f1(x) and f2(x)')
    axis.grid()
    axis.plot(X, Y, label='f(x)')
    axis.plot(X, dY, label='x')

    plt.show()

def check_phi(a, b):
    data = [('x', 'phi(x)', 'dphi(x)')]
    for x in np.linspace(a, b, 10):
        data.append((x, phi(x), dphi(x)))
    return data

def main(): 
    # ln(x+2) - x^4 + 0.5

    ab = [1, 1.2]
    x0 = 1.15

    phi_data = check_phi(ab[0], ab[1])

    print("[{}, {}]".format(ab[0], ab[1]))
    print(tabulate(phi_data, headers='firstrow'))
    print("\n")

    iterations_data = simple_iterations(phi, dphi, ab[0], ab[1])
    print("Simple iteration:")
    print(tabulate(iterations_data, headers='firstrow'))
    print("\n")

    newton_data = newton(f, df, x0)
    print("Newton:")
    print(tabulate(newton_data, headers='firstrow'))
    print("\n")

    plot(f1, f2, [0, 2], step=0.1)

if __name__ == '__main__':
    main()
