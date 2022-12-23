# X* = 2.0
# i:         0    1        2         3       4
# Xi:        1   1.5      2.0       2.5     3.0
# Yi:       2.0  2.1344  2.4702   2.9506  3.5486

def find_interval(x, x0):
    for i in range(len(x) - 1):
        if x[i] <= x0 <= x[i + 1]:
            return i

def first_derivative(x, y, x0):
    i = find_interval(x, x0)
    derivative = (y[i + 1] - y[i - 1])/(x[i + 1] - x[i - 1])
    return derivative

def second_derivative(x, y, x0):
    i = find_interval(x, x0)
    derivative2 = (y[i - 1] - 2*y[i] + y[i + 1])/(x[i + 1] - x[i])**2
    return derivative2

def main():
    x0 = 1.0
    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [1.0, 1.3776, 1.5403, 1.5707, 1.5839]
    print(f'First derivative = {first_derivative(x, y, x0)}')
    print(f'Second derivative = {second_derivative(x, y, x0)}')

if __name__ == '__main__':
    main()