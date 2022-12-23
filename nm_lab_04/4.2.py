
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from progonka import tma

PI = np.pi

def RungeKutta(f, a, b, h, y0, y_der):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    k = [y_der]
    for i in range(n):
        K1 = h * g(x[i], y[i], k[i])
        L1 = h * f(x[i], y[i], k[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, k[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, k[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, k[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, k[i] + L3)
        y.append(y[i] + (K1 + 2 * K2 + 2 * K3 + K4) / 6)
        k.append(k[i] + (L1 + 2 * L2 + 2 * L3 + L4) / 6)
    return x, y, k

def func(x, y, y_der):
    return 2 * (1 + (np.tan(x) ** 2)) * y

def g(x, y, k):
    return k

def p(x):
    return 0

def q(x):
    return - 2 * (1 + (np.tan(x) ** 2))

def exactFunction(x):
    return -np.tan(x)

def stop(y, y1, eps):
    if abs(y[-1] - y1) > eps:
        return True
    else:
        return False

def eta_next(eta_prev, eta, ans_prev, ans, b, y1):
    _, y = ans_prev[0], ans_prev[1]
    phi_last = y[-1] - y1
    _, y = ans[0], ans[1]
    phi = y[-1] - y1
    return eta - (eta - eta_prev) / (phi - phi_last) * phi

def shooting_method(a, b, y0, y1, h, eps):
    eta_prev = 1
    eta = 0.8
    y_der = eta_prev
    ans_prev = RungeKutta(func, a, b, h, eta_prev, y_der)[:2]
    y_der = eta
    ans = RungeKutta(func, a, b, h, eta, y_der)[:2]

    while stop(ans[1], y1, eps):
        eta, eta_prev = eta_next(eta_prev, eta, ans_prev, ans, b, y1), eta
        ans_prev = ans
        y_der = eta
        ans = RungeKutta(func, a, b, h, y0, y_der)[:2]

    return ans

def finite_difference_method(a, b, alpha, beta, delta, gamma, y0, y1, h):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    A = [0] + [1 - p(x[i]) * h / 2 for i in range(0, n - 1)] + [-gamma]
    B = [alpha * h - beta] + [q(x[i]) * h ** 2 - 2 for i in range(0, n - 1)] + [delta * h + gamma]

    C = [beta] + [1 + p(x[i]) * h / 2 for i in range(0, n - 1)] + [0]
    D = [y0 * h] + [p(x[i]) * h ** 2 for i in range(0, n - 1)] + [y1 * h]

    y = tma(A, B, C, D)
    return x, y

def Runge_romberg(ans):
    Y1 = [yi for xi, yi in zip(ans[0]['Shooting']['x'], ans[0]['Shooting']['y']) if xi in ans[1]['Shooting']['x']]
    Y2 = [yi for xi, yi in zip(ans[1]['Shooting']['x'], ans[1]['Shooting']['y']) if xi in ans[0]['Shooting']['x']]
    shoot_err = [(y2 - y1) / (2 ** 4 - 1) for y1, y2 in zip(Y1, Y2)]

    Y1 = [yi for xi, yi in zip(ans[0]['FD']['x'], ans[0]['FD']['y']) if xi in ans[1]['FD']['x']]
    Y2 = [yi for xi, yi in zip(ans[1]['FD']['x'], ans[1]['FD']['y']) if xi in ans[0]['FD']['x']]
    fd_err = [(y2 - y1) / (2 ** 4 - 1) for y1, y2 in zip(Y1, Y2)]

    return {'Shooting': shoot_err, 'FD': fd_err}

def print_points(_x, _y):
    data = [('x', 'y')]
    for x, y in zip(_x, _y):
        data.append((x, y))
    print(tabulate(data, headers='firstrow'), '\n')

def show_plot(ans, exact, h):
    n = len(ans)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.6)
        plt.scatter(ans[i]["Shooting"]["x"], ans[i]["Shooting"]["y"], color='r', alpha=0.4, label='Shooting method')
        plt.plot(ans[i]["Shooting"]["x"], ans[i]["Shooting"]["y"], color='r', alpha=0.4)
        plt.scatter(ans[i]["FD"]["x"], ans[i]["FD"]["y"], color='b', alpha=0.4, label='Finite difference method')
        plt.plot(ans[i]["FD"]["x"], ans[i]["FD"]["y"], color='b', alpha=0.4)
        plt.scatter(exact[i][0], exact[i][1], color='g', alpha=0.4, label='Exact solution')
        plt.plot(exact[i][0], exact[i][1], color='g', alpha=0.4)

        plt.legend(loc='best')
        plt.title('h{0} = '.format(i + 1) + str(h[i]))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    plt.show()

def main():
    a = 0
    b = PI / 6
    alpha = 1
    delta = 1
    gamma = 0
    beta = 0
    y0 = 0.0
    y1 = exactFunction(b)
    step = PI / 30
    eps = 1e-5

    print(f'Interval: [{a}, {b}]')
    print(f'y0 = {y0}, y1 = {y1}')
    print()

    res = []
    res2 = []
    ans = []
    steps = [step, step / 2]
    i = 0

    for h in steps:
        print(f'Current step: {h}')
        print('Shooting method')
        res.append(shooting_method(a, b, y0, y1, h, eps))
        for x, y in zip(res[i][0], res[i][1]):
            print(f'x: {round(x, 5)}, y: {round(y, 5)}')
        print()

        print('Finite difference method')
        res2.append(finite_difference_method(a, b, alpha, beta, delta, gamma, y0, y1, h))
        for x, y in zip(res2[i][0], res2[i][1]):
            print(f'x: {round(x, 5)}, y: {round(y, 5)}')
        print()

        ans.append({
            "h": h,
            "Shooting": {'x': res[i][0],  'y': res[i][1]},
            "FD":       {'x': res2[i][0], 'y': res2[i][1]}
        })

        i += 1

    exact = []
    for h in steps:
        x_ex = [i for i in np.arange(a, b + h, h)]
        y_ex = [exactFunction(i) for i in x_ex]
        exact.append((x_ex, y_ex))

    err = Runge_romberg(ans)
    print('Shooting method RR error: {}'.format(err['Shooting']))
    print('Finite difference method RR error: {}'.format(err['FD']))
    show_plot(ans, exact, steps)
    
if __name__ == '__main__':
    main()