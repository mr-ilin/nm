import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def func(x, y, y_der):
    return -4 * x * y_der - (4 * x ** 2 + 2) * y

def exactFunction(x):
    return (1 + x) * np.exp(-x ** 2)

def g(x, y, k):
    return k

def analytical(f, a, b, h):
    x = [i for i in np.arange(a, b + h, h)]
    y = [f(i) for i in x]
    return x, y

def Euler(f, a, b, h, y0, y_der):
    n = int((b - a) / h)
    x = [i for i in np.arange(a, b + h, h)]
    y = [y0]
    k = y_der
    for i in range(n):
        k += h * f(x[i], y[i], k)
        y.append(y[i] + h * g(x[i], y[i], k))

    return x, y

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

def Adams(f, x, y, k, h):
    n = len(x)
    x = x[:4]
    y = y[:4]
    k = k[:4]
    for i in range(3, n - 1):
        k.append(k[i] + h * (55 * f(x[i], y[i], k[i]) - 59 * f(x[i - 1], y[i - 1], k[i - 1]) + 37 * f(x[i - 2], y[i - 2], k[i - 2]) - 9 * f(x[i - 3], y[i - 3], k[i - 3])) / 24)
        y.append(y[i] + h * (55 * g(x[i], y[i], k[i]) - 59 * g(x[i - 1], y[i - 1], k[i - 1]) + 37 * g(x[i - 2], y[i - 2], k[i - 2]) - 9 * g(x[i - 3], y[i - 3], k[i - 3])) / 24)
        x.append(x[i] + h)
    return x, y


def RungeRomberg(dict):
    Y1 = [yi for xi, yi in zip(dict[0]['Euler']['x'], dict[0]['Euler']['y']) if xi in dict[1]['Euler']['x']]
    Y2 = [yi for xi, yi in zip(dict[1]['Euler']['x'], dict[1]['Euler']['y']) if xi in dict[0]['Euler']['x']]
    Euler = [(y2 - y1) / (2 ** 1 - 1) for y1, y2 in zip(Y1, Y2)]

    Y1 = [yi for xi, yi in zip(dict[0]['Runge']['x'], dict[0]['Runge']['y']) if xi in dict[1]['Runge']['x']]
    Y2 = [yi for xi, yi in zip(dict[1]['Runge']['x'], dict[1]['Runge']['y']) if xi in dict[0]['Runge']['x']]
    runge = [(y2 - y1) / (2 ** 4 - 1) for y1, y2 in zip(Y1, Y2)]

    Y1 = [yi for xi, yi in zip(dict[0]['Adams']['x'], dict[0]['Adams']['y']) if xi in dict[1]['Adams']['x']]
    Y2 = [yi for xi, yi in zip(dict[1]['Adams']['x'], dict[1]['Adams']['y']) if xi in dict[0]['Adams']['x']]
    Adams = [(y2 - y1) / (2 ** 4 - 1) for y1, y2 in zip(Y1, Y2)]

    return {'Euler': Euler, 'Runge': runge, 'Adams': Adams}


def show_plot(res, exact, h):
    n = len(res)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.subplots_adjust(wspace=0.1, hspace=0.6)
        plt.scatter(res[i]["Euler"]["x"], res[i]["Euler"]["y"], color='r', alpha=0.4, label='Euler method')
        plt.plot(res[i]["Euler"]["x"], res[i]["Euler"]["y"], color='r', alpha=0.4)
        plt.scatter(res[i]["Runge"]["x"], res[i]["Runge"]["y"], color='b', alpha=0.4, label='Runge-Kutta method')
        plt.plot(res[i]["Runge"]["x"], res[i]["Runge"]["y"], color='b', alpha=0.4)
        plt.scatter(res[i]["Adams"]["x"], res[i]["Adams"]["y"], color='g', alpha=0.4, label='Adams method')
        plt.plot(res[i]["Adams"]["x"], res[i]["Adams"]["y"], color='g', alpha=0.4)
        plt.scatter(exact[i][0], exact[i][1], color='k', alpha=0.4, label='exact function')
        plt.plot(exact[i][0], exact[i][1], color='k', alpha=0.4)

        plt.legend(loc='best')
        plt.title('h{0} = '.format(i + 1) + str(h[i]))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    plt.show()

def print_points(_x, _y):
    data = [('x', 'y')]
    for x, y in zip(_x, _y):
        data.append((x, y))
    print(tabulate(data, headers='firstrow'), '\n')


def main():
    a = 0
    b = 1
    h = 0.1
    y0 = 1
    y_der = 1
    res = []
    exact = []
    steps = [h, h/2]

    for h in steps:
        print(f"Step: {h}")
        print("Euler:")
        x_euler, y_euler = Euler(func, a, b, h, y0, y_der)
        print_points(x_euler, y_euler)

        print("Runge-Kutta:")
        x_runge, y_runge, k_runge = RungeKutta(func, a, b, h, y0, y_der)
        print_points(x_runge, y_runge)

        print("Adams:")
        x_adams, y_adams = Adams(func, x_runge, y_runge, k_runge, h)
        print_points(x_adams, y_adams)

        print("Analytical:")
        x_analytical, y_analytical = analytical(exactFunction, a, b, h)
        print_points(x_analytical, y_analytical)

        exact.append((x_analytical, y_analytical))
        res.append({
            "h": h,
            "Euler": {'x': x_euler, 'y': y_euler},
            "Runge": {'x': x_runge, 'y': y_runge},
            "Adams": {'x': x_adams, 'y': y_adams},
        })

    err = RungeRomberg(res)
    print("Euler RR error: {0}".format(err['Euler']))
    print("Runge RR error: {0}".format(err['Runge']))
    print("Adams RR error: {0}".format(err['Adams']))

    show_plot(res, exact, steps)

if __name__ == '__main__':
    main()
