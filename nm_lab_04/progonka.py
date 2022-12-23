def tridiag(matrix, n, d):
    size = int(n)
    a = []
    b = []
    c = []
    cnt = 0
    for i in range(size):
        for j in range(size):
            if i == j:
                b.append(float(matrix[i][j]))
        if (cnt != size - 1):
            c.append(float(matrix[cnt][cnt + 1]))
            a.append(float(matrix[cnt + 1][cnt]))
            cnt += 1

    y = [0 for i in range(size)]
    alpha = [0 for i in range(size)]
    beta = [0 for i in range(size)]

    y[0] = b[0]
    alpha[0] = -c[0] / y[0]
    beta[0]  = d[0] / y[0]

    # first
    for i in range(1, size):
        y[i] = b[i] + alpha[i - 1] * a[i - 1]
        if i != size -1:
            alpha[i] = -c[i] / y[i]
        beta[i] = (d[i] - beta[i - 1] * a[i -1]) / y[i]

    # Reverse
    x = [0 for i in range(n)]
    x[-1] = beta[-1]
    for i in range(size - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x

def tma(a, b, c, d):
    size = len(a)
    p, q = [], []
    p.append(-c[0] / b[0])
    q.append(d[0] / b[0])

    for i in range(1, size):
        p_tmp = -c[i] / (b[i] + a[i] * p[i - 1])
        q_tmp = (d[i] - a[i] * q[i - 1]) / (b[i] + a[i] * p[i - 1])
        p.append(p_tmp)
        q.append(q_tmp)

    x = [0 for _ in range(size)]
    x[size - 1] = q[size - 1]

    for i in range(size - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return x

def outp(A, n):
    print("Start matrix")
    for i in range(n):
        for j in range(n):
            print("\t", A[i][j], " ", end = "")
        print("\n")

if __name__ == '__main__':
    n = int(input())
    matrix = []
    for i in range(n):
        matrix.append(list(map(float, input().split())))

    d = list(map(float, input().split()))

    outp(matrix, n)
    tridiag(matrix, n, d)