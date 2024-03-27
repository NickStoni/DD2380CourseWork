import math
import time

TIME_LIMIT = 1

def flatten(A):
    v = []
    for row in A:
        for e in row:
            v.append(e)
    return v


def matrix_mult(A, B):
    # The number of columns in A has to be the same as the number of rows in B, in order for AB to be defined
    if len(A[0]) != len(B):
        raise Exception("The dimensions of A and B do not match!")

    C = [[0] * len(B[0]) for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            c_acc = 0
            for k in range(len(A[0])):
                c_acc += A[i][k] * B[k][j]
            C[i][j] = c_acc
    return C


def transform_flattened_to_matrix(A, n, k):
    B = [[0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            B[i][j] = A[i * k + j]
    return B


def calc_scaled_alpha(A, B, pi, O):
    N = len(A)
    T = len(O)

    alpha = [[0]*N for _ in range(T)]
    c = [0]*T

    # Get alpha_1
    for i in range(N):
        alpha[0][i] = B[i][O[0]]*pi[i]
        c[0] += alpha[0][i]

    # Scale alpha_1
    alpha[0] = [alpha[0][i]/c[0] for i in range(len(alpha[0]))]

    for t in range(1, T):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += A[j][i]*alpha[t-1][j]
            alpha[t][i] = B[i][O[t]]*temp
            c[t] += alpha[t][i]

        alpha[t] = [alpha[t][i]/c[t] for i in range(len(alpha[t]))]
    return alpha, c


def calc_scaled_beta(A, B, O, c):
    N = len(A)
    T = len(O)

    beta = [[1/c[-1]]*N for _ in range(T)]

    for t in range(T-2, -1, -1):
        for i in range(N):
            temp_beta = 0
            for j in range(N):
                temp_beta += beta[t+1][j]*B[j][O[t+1]]*A[i][j]
            beta[t][i] = temp_beta/c[t]
    return beta


def calc_gamma(A, B, O, alpha, beta):
    N = len(A)
    T = len(O)

    di_gamma = [[[0 for _ in range(N)] for _ in range(N)] for _ in range(T)]
    gamma = [[0 for _ in range(N)] for _ in range(T)]

    for t in range(T - 1):
        for i in range(N):
            for j in range(N):
                di_gamma[t][i][j] = alpha[t][i] * A[i][j] * B[j][O[t + 1]] * beta[t + 1][j]

        for i in range(N):
            gamma[t][i] = sum(di_gamma[t][i])

    for i in range(N):
        gamma[T-1][i] = alpha[T-1][i]

    return di_gamma, gamma


def estimate_lambda(A, B, pi, O):
    N = len(A)
    T = len(O)
    K = len(B[0])

    alpha, c = calc_scaled_alpha(A, B, pi, O)
    beta = calc_scaled_beta(A, B, O, c)

    di_gamma, gamma = calc_gamma(A, B, O, alpha, beta)
    A_upd = [[0 for _ in range(N)] for _ in range(N)]
    B_upd = [[0 for _ in range(K)] for _ in range(N)]
    pi_upd = [0]*len(pi)

    for i in range(N):
        pi_upd[i] = gamma[0][i]

    for i in range(N):
        sum_gamma = 0
        for t in range(T - 1):
            sum_gamma += gamma[t][i]

        for j in range(N):
            sum_di_gamma = 0
            for t in range(T - 1):
                sum_di_gamma += di_gamma[t][i][j]
            A_upd[i][j] = sum_di_gamma / sum_gamma

    for j in range(N):
        sum_gamma = 0
        for t in range(T):
            sum_gamma += gamma[t][j]

        for k in range(K):
            sum_indicator = 0
            for t in range(T):
                sum_indicator += indicator(O[t], k)*gamma[t][j]
            B_upd[j][k] = sum_indicator / sum_gamma

    return A_upd, B_upd, pi_upd, c


def indicator(a, b):
    return 1 if a == b else 0

def do_baum_welch(A, B, pi, O, precision, START_TIME):

    oldLogPorb = -math.inf
    converged = False
    it = 0
    while not converged and time.time() - START_TIME < TIME_LIMIT*0.85:
        A, B, pi, c = estimate_lambda(A, B, pi, O)

        it += 1

        logProb = -sum(math.log(1/round(c_i, precision+1)) for c_i in c)
        if logProb <= oldLogPorb:
            converged = True

        oldLogPorb = logProb
    return A, B, pi


def stringify_matrix(A, precision):
    n = len(A)
    k = len(A[0])

    flattened_A = flatten(A)
    flattened_A.insert(0, k)
    flattened_A.insert(0, n)

    return " ".join(str(round(e, precision)) for e in flattened_A)

def estimate_model_parameters():
    precision = 7
    START_TIME = time.time()

    A_guess_flat = [float(x) for x in input().split()[2:]]
    B_guess_flat = [float(x) for x in input().split()[2:]]
    pi_guess = [float(x) for x in input().split()[2:]]
    O = [int(x) for x in input().split()[1:]]

    n = int(math.sqrt(len(A_guess_flat)))
    k = int(len(B_guess_flat) / n)

    A = transform_flattened_to_matrix(A_guess_flat, n, n)
    B = transform_flattened_to_matrix(B_guess_flat, n, k)

    A, B, pi = do_baum_welch(A, B, pi_guess, O, precision, START_TIME)
    return stringify_matrix(A, precision), stringify_matrix(B, precision)


if __name__ == "__main__":
    stringified_A, stringified_B = estimate_model_parameters()
    print(stringified_A)
    print(stringified_B)
