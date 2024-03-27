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


def flatten(A):
    v = []
    for row in A:
        for e in row:
            v.append(e)
    return v


def convert_matrix_to_str(A):
    n = len(A)
    k = len(A[0])
    flattened_A = flatten(A)

    # A work-around to add dimensions to the final result
    flattened_A.insert(0, k)
    flattened_A.insert(0, n)

    return " ".join(str(round(e, 6)) for e in flattened_A)  # Round is needed because float is imprecise and might give
    # weird output


def calc_alpha(A, B, pi, O):
    N = len(A)
    T = len(O)

    alpha = [[0]*N for _ in range(T)]

    # Get alpha_1
    for i in range(N):
        alpha[0][i] = B[i][O[0]]*pi[i]

    for t in range(1,T):
        for i in range(N):
            temp = 0
            for j in range(N):
                temp += A[j][i]*alpha[t-1][j]
            alpha[t][i] = B[i][O[t]]*temp
    return alpha

def sequence_probability():
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

    alpha = calc_alpha(A, B, pi, O)

    return sum(alpha[len(O)-1])


if __name__ == "__main__":
    print(round(sequence_probability(), 6))
