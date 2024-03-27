import math


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


def calc_delta(A, B, pi, O):
    N = len(A)
    T = len(O)

    delta = [[0] * N for _ in range(T)]
    delta_idx = [[0] * N for _ in range(T)]

    # Get delta_1
    for i in range(N):
        delta[0][i] = B[i][O[0]] * pi[i]

    for t in range(1, T):
        for i in range(N):
            for j in range(N):
                temp_delta = A[j][i] * delta[t - 1][j] * B[i][O[t]]
                if temp_delta > delta[t][i]:
                    delta[t][i] = temp_delta
                    delta_idx[t][i] = j

    return delta, delta_idx


def most_likely_sequence():
    A_flat = [float(x) for x in input().split()[2:]]  # Discard the first time numbers as they only tell us dimensions
    # which is redundant information as it can be obtained from the
    # length of the given vector, applies to A, B and pi.
    B_flat = [float(x) for x in input().split()[2:]]
    pi = [float(x) for x in input().split()[2:]]
    O = [int(x) for x in input().split()[1:]]

    n = int(math.sqrt(len(A_flat)))
    k = int(len(B_flat) / n)

    A = transform_flattened_to_matrix(A_flat, n, n)
    B = transform_flattened_to_matrix(B_flat, n, k)

    delta, delta_idx = calc_delta(A, B, pi, O)

    sequence = [0]*len(O)

    # Calculate the most likely last state
    sequence[-1] = delta[len(O)-1].index(max(delta[len(O)-1]))

    # Backward iteratively obtain the preceding most likely states
    for t in range(len(O)-2, -1, -1):
        sequence[t] = delta_idx[t+1][sequence[t+1]]

    return ' '.join(str(e) for e in sequence)


if __name__ == "__main__":
    print(most_likely_sequence())
