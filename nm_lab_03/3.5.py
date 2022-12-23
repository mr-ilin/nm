import numpy as np

def function(x):
    return 1 / (x**3 + 64)

def count_x(x0, x, step):
    return [i for i in np.arange(x0, x + step, step)]

def count_y(x):
    return [function(i) for i in x]

def rectangle(x, h):
    return h * sum([function((x[i] + x[i + 1]) / 2) for i in range(len(x) - 1)])

def trapeze(x, h):
    y = count_y(x)
    return h * (y[0] / 2 + sum([y[i] for i in range(1, len(y) - 2)]) + y[len(y) - 1] / 2)

def simpson(x, h):
    y = count_y(x)
    return h / 3 * (
        y[0] +
        sum([4 * y[i] for i in range(1, len(y) - 1, 2)]) +
        sum([2 * y[i] for i in range(2, len(y) - 2, 2)]) +
        y[len(y) - 1]
    )

def runge_romberg_error(res):
    error_rectangle = abs((res[0]['rectangle']  - res[1]['rectangle'])) / (2 ** 2 - 1)
    error_trapeze = abs((res[0]['trapeze']  - res[1]['trapeze'])) / (2 ** 2 - 1)
    error_simpson = abs((res[0]['simpson']  - res[1]['simpson'])) / (2 ** 4 - 1)

    return {'rectangle': error_rectangle, 'trapeze': error_trapeze, 'simpson': error_simpson}

def main():
    x0 = -2
    x = 2
    steps = [0.5, 1]
    true_value = 0.0626406953808367
    result = []
    print(f'True value of integral = {true_value}\n')

    for h in steps:
        print('h =', h)
        X = count_x(x0, x, h)
        print(f'x = {x}')
        Y = count_y(X)
        print(f'y = {Y}')

        res_rectangle = rectangle(X, h)
        print('Rectangle method:', res_rectangle, '\n')

        res_trapeze = trapeze(X, h)
        print('Trapeze method:', res_trapeze, '\n')

        res_simpson = simpson(X, h)
        print('Simpson method:', res_simpson, '\n')

        result.append({
            'h': h,
            'rectangle':res_rectangle,
            'trapeze': res_trapeze,
            'simpson': res_simpson
        })

    err = runge_romberg_error(result)

    print('rectangle RR error: {}'.format(err['rectangle']))
    print('trapeze RR error {}'.format(err['trapeze']))
    print('Simpson RR error {}'.format(err['simpson']))

if __name__ == '__main__':
    main()