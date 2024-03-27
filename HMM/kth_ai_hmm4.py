#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import time
import math

import random

MARGIN = 20
N_GUESSES = N_SPECIES
START_TRAINING_AT = N_STEPS - N_FISH + N_GUESSES - MARGIN
N_FEATURES = 5  # This proved to be a good number of features
DUMMY_VALUE = -1234
TIME_OUT_THRESHOLD = 0.95


class MatrixUtilities:
    """
    A helper static class containing different matrix utility functions
    """

    @staticmethod
    def generate_initial_matrix(n, k):
        # Initializes almost uniform distribution matrix of n rows and k columns
        A = []
        for i in range(n):
            A.append(MatrixUtilities.generate_initial_guess_vec(k))
        return A

    @staticmethod
    def matrix_mult(A, B):
        # Standard implementation of matrix multiplication
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

    @staticmethod
    def first_non_zero_decimal(number):
        # To avoid exceptions when taking log(0)
        if number == 0:
            return -1

        # Ingenious formula to get the first non-zero digit.
        return math.ceil(-math.log10(abs(number) - abs(math.floor(number))))

    @staticmethod
    def generate_initial_guess_vec(k):
        # Do not want exactly uniform distribution, so added ability to adjust the weights for even and odd entries
        WEIGHT_M_1 = 1.2
        WEIGHT_M_2 = 2.0 - WEIGHT_M_1

        # Modified, so-called almost uniform distribution with custom weights.
        m_1 = (1 / k) * WEIGHT_M_1
        m_2 = (1 / k) * WEIGHT_M_2

        v = [0] * k

        acc = 0
        for i in range(k):
            if i == k - 1:
                # This is needed since we want columns to add up to 1
                v[i] = 1.0 - acc
            else:
                # Give odd and even entries different weighed values
                v[i] = m_1 if i % 2 else m_2
                acc += v[i]
        return v


class HMM:
    """
    A class representing HMM, containing all the necessary functions, such as train_HMM to train HMM on given
    observations and get_probability_of_observations to return the probability of having observed a sequence.
    Implementation for the training pass was taken from Mark Stamp tutorial.
    """

    def __init__(self, n_states, n_emissions):
        self.N = n_states
        self.K = n_emissions
        self.T = 0

        self.A = MatrixUtilities.generate_initial_matrix(n_states, n_states)
        self.B = MatrixUtilities.generate_initial_matrix(n_states, n_emissions)
        self.pi = MatrixUtilities.generate_initial_guess_vec(n_states)
        self.O = []

    def train_HMM(self, O, start_time):
        # The implementation given by Mark Stamp in his Introduction to Hidden Markov Models

        self.O = O
        self.T = len(self.O)

        old_log_prob = -math.inf
        converged = False
        it = 0

        # Do the training until c values do not increase anymore or if we have run out of time
        while not converged and time.time() - start_time < STEP_TIME_THRESHOLD * TIME_OUT_THRESHOLD:
            self.A, self.B, self.pi, c = self.estimate_lambda()

            it += 1

            log_prob = -sum(math.log(1 / c_i) for c_i in c)

            # Stop training if the probability is not maximized anymore
            if log_prob <= old_log_prob:
                converged = True

            old_log_prob = log_prob

    def get_probability_of_observations(self, O):
        # Implementation of alpha (forward) pass, to get probability of having observed O, given the model.
        # Do not scale alpha here, as the probability will be compared to that of other models with
        # other scaling factors. Also saves time.

        T = len(O)  # Need a new T, since len(O) will not match the length of the dataset that HMM was trained at.
        alpha = [[0] * self.N for _ in range(T)]

        # Get alpha_1
        for i in range(self.N):
            alpha[0][i] = self.B[i][O[0]] * self.pi[i]

        # Get alpha_2 through alpha_T
        for t in range(1, T):
            for i in range(self.N):
                temp = 0
                for j in range(self.N):
                    temp += self.A[j][i] * alpha[t - 1][j]
                alpha[t][i] = self.B[i][O[t]] * temp

        # Sum of all alphas at T gives the probability.
        return sum(alpha[T - 1])

    def calc_scaled_alpha(self):
        # Alpha pass function for training HMM, here we scale alphas to avoid underflow of float, also c is used
        # to calculate if the probability after the pass increased.

        alpha = [[0] * self.N for _ in range(self.T)]
        c = [0] * self.T

        # Get alpha_1
        for i in range(self.N):
            alpha[0][i] = self.B[i][self.O[0]] * self.pi[i]
            c[0] += alpha[0][i]

        # Scale alpha_1
        alpha[0] = [alpha[0][i] / c[0] for i in range(len(alpha[0]))]

        # Get alpha_2 through alpha_T
        for t in range(1, self.T):
            for i in range(self.N):
                temp = 0
                for j in range(self.N):
                    temp += self.A[j][i] * alpha[t - 1][j]
                alpha[t][i] = self.B[i][self.O[t]] * temp
                c[t] += alpha[t][i]

            # Scale alpha_t
            alpha[t] = [alpha[t][i] / c[t] for i in range(len(alpha[t]))]

        return alpha, c

    def calc_scaled_beta(self, c):
        # Beta (backward) pass function for training HMM, here we scale betas to avoid underflow of float.

        # Initialise beta as 1 over the last scaling factor (since it is a backward pass algorithm)
        beta = [[1 / c[-1]] * self.N for _ in range(self.T)]

        # Calculate beta_t-1 through beta_1
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                temp_beta = 0
                for j in range(self.N):
                    temp_beta += beta[t + 1][j] * self.B[j][self.O[t + 1]] * self.A[i][j]

                # Scale beta_t
                beta[t][i] = temp_beta / c[t]
        return beta

    def calc_gamma(self, alpha, beta):
        # Function that calculates gamma and di-gamma given alpha and beta. There is no reason to use scaling for them,
        # as alpha and beta are already scaled.

        di_gamma = [[[0 for _ in range(self.N)] for _ in range(self.N)] for _ in range(self.T)]
        gamma = [[0 for _ in range(self.N)] for _ in range(self.T)]

        # Calculate di-gamma and gamma as described in Mark Stamp's tutorial
        for t in range(self.T - 1):
            for i in range(self.N):
                for j in range(self.N):
                    di_gamma[t][i][j] = alpha[t][i] * self.A[i][j] * self.B[j][self.O[t + 1]] * beta[t + 1][j]

            for i in range(self.N):
                gamma[t][i] = sum(di_gamma[t][i])

        for i in range(self.N):
            gamma[self.T - 1][i] = alpha[self.T - 1][i]

        return di_gamma, gamma

    def estimate_lambda(self):
        # The function that does one iteration of Baum-Welch algorithm

        alpha, c = self.calc_scaled_alpha()
        beta = self.calc_scaled_beta(c)

        di_gamma, gamma = self.calc_gamma(alpha, beta)

        # Do not want to mess around with mutable types in python, better initialize new lists
        A_upd = [[0 for _ in range(self.N)] for _ in range(self.N)]
        B_upd = [[0 for _ in range(self.K)] for _ in range(self.N)]
        pi_upd = [0] * len(self.pi)

        # Update pi vector (describes initial hidden state distribution)
        for i in range(self.N):
            pi_upd[i] = gamma[0][i]

        # Update A matrix (describes probabilities of transferring from one hidden state to another)
        for i in range(self.N):
            sum_gamma = 0
            for t in range(self.T - 1):
                sum_gamma += gamma[t][i]

            for j in range(self.N):
                sum_di_gamma = 0
                for t in range(self.T - 1):
                    sum_di_gamma += di_gamma[t][i][j]
                A_upd[i][j] = sum_di_gamma / sum_gamma

        # Update B matrix (describes probabilities of observing an emission given hidden state)
        for j in range(self.N):
            sum_gamma = 0
            for t in range(self.T):
                sum_gamma += gamma[t][j]

            for k in range(self.K):
                sum_indicator = 0
                for t in range(self.T):
                    sum_indicator += self.indicator(self.O[t], k) * gamma[t][j]
                B_upd[j][k] = sum_indicator / sum_gamma

        return A_upd, B_upd, pi_upd, c

    def indicator(self, a, b):
        # The so-called indicator function, which is one if arguments are equal and 0 otherwise.
        return 1 if a == b else 0


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        We initialise O used for observations for each fish index,
        known_fish containing all indexes that we have made a guess for,
        known_types to store the mapping between fish index and fish type for fishes that we have made guesses for,
        trained_HMM to store the trained HMM's for different fish types,
        to_retrain stores what fish type that respective HMM failed for. We want to retrain this HMM in the next
        iteration.
        """
        self.O = dict()
        self.known_fish = set()
        self.known_types = dict()
        self.trained_HMM = dict()

        self.to_retrain = DUMMY_VALUE

    def guess(self, step, observations):
        """
        This method gets called on every iteration, providing observations.
        Here the player should process and store this information,
        and optionally make a guess by returning a tuple containing the fish index and the guess.
        :param step: iteration number
        :param observations: a list of N_FISH observations, encoded as integers
        :return: None or a tuple (fish_id, fish_type)
        """

        # Here is the strategy we are using: Create a separate HMM for each fish type, make a couple guesses and try
        # to encounter as many fish types as possible. Then store the mappings between the fish index and fish type
        # and when enough time has passed, train HMM's on the known fish types and start guessing.

        iteration_start_time = time.time()
        self.append_observations(observations)

        # Want to guess a couple fish before we start making guesses, how can we otherwise train HMM's? (rhetorical q)
        # But not necessary to make a guess if we already encountered at least one fish of each type.
        if step < N_GUESSES and len(self.trained_HMM) < N_SPECIES:
            fish_index = random.randint(0, N_FISH - 1)
            while fish_index in self.known_fish:
                fish_index = random.randint(0, N_FISH - 1)
            return (fish_index, random.randint(0, N_SPECIES - 1))

        # The waiting until enough time has passed step!
        if step < START_TRAINING_AT:
            return None

        # Retain the model that failed last time!
        if self.to_retrain != DUMMY_VALUE:
            if self.to_retrain not in self.trained_HMM:
                raise Exception("to_retrain variable has not been initialized correctly!")

            self.add_HMM(self.to_retrain, iteration_start_time)
            self.to_retrain = DUMMY_VALUE

        # Check if all the collected data and revealed states up to this point have been used to train HMM
        if len(self.trained_HMM) < len(self.known_types):
            for type in self.known_types:
                if type not in self.trained_HMM:
                    # Train only one HMM at a time
                    # Do not want to start making guesses until all the models have been trained.
                    self.add_HMM(type, iteration_start_time)
                    return None

            raise Exception("Known fish types dictionary has not been initialized correctly!")

        # Wanted to test random order of indexes and its effect on performance.
        shuffled_indexes = [x for x in range(len(observations))]
        random.shuffle(shuffled_indexes)
        for fish_index in shuffled_indexes:
            # For the first never encountered fish index, test it against all trained HMM's and return the fish type
            # that has the highest probability, given the observations.
            if fish_index not in self.known_fish:
                most_likely_species_index = -1
                highest_probability = -1
                for HMM_index in self.trained_HMM:
                    current_probability = self.trained_HMM[HMM_index].get_probability_of_observations(
                        self.O[fish_index])
                    if current_probability > highest_probability:
                        highest_probability = current_probability
                        most_likely_species_index = HMM_index

                return (fish_index, most_likely_species_index)

    def add_HMM(self, fish_type, start_time):
        # The function that initializes the training of an HMM and adds it to the dictionary containing all HMM's
        id_of_known_fish_of_type = self.known_types[fish_type][0]
        new_HMM = HMM(N_FEATURES, N_EMISSIONS)
        new_HMM.train_HMM(self.O[id_of_known_fish_of_type], start_time)
        self.trained_HMM[fish_type] = new_HMM

    def append_observations(self, observations):
        # The function that adds all new observations for each fish to the dictionary containing observations for all
        # fish up until the current time stamp.
        for fish_index in range(len(observations)):
            if fish_index not in self.O:
                self.O[fish_index] = []
            self.O[fish_index].append(observations[fish_index])

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        # Want to retrain the failed model!
        if not correct and true_type in self.trained_HMM:
            self.to_retrain = true_type

        # Want to memorize the encountered fish and map their index to their type.
        self.known_fish.add(fish_id)
        if true_type not in self.known_types:
            self.known_types[true_type] = []

        self.known_types[true_type].append(fish_id)
