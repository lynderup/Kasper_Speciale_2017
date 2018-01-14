import numpy as np


def observables_to_indexs(observables, list_of_observables):
    return [observables.index(state) for state in list_of_observables]


class HMM:

    def __init__(self, hidden_states, observables, initials, transitions, emissions, hidden_to_annotations):

        self.hidden_states = hidden_states
        self.observables = observables

        self.initials = initials
        self.transitions = transitions
        self.emissions = emissions

        self.log_initials = np.asarray([-np.inf if prop == 0 else np.log(prop) for prop in initials])
        self.log_transitions = np.asarray([[-np.inf if prop == 0 else np.log(prop)
                                            for prop in row] for row in transitions])
        self.log_emissions = np.log(emissions)

        self.hidden_to_annotations = hidden_to_annotations

    def viterbi_decoding(self, xs):
        omega = self.compute_omega(xs)
        return self.backtrack(omega, xs)

    def compute_omega(self, xs):

        xs = observables_to_indexs(self.observables, xs)

        N = len(xs)
        K = len(self.hidden_states)

        omega = np.full((K, N), -np.inf)

        for k in range(K):
            omega[k, 0] = self.log_initials[k] + self.log_emissions[k, xs[0]]

        for n in range(1, N):
            for k in range(K):
                prop_of_most_likely_old_z = np.max([self.log_transitions[z, k] + old_omega for z, old_omega in enumerate(omega[:, n - 1])])
                omega[k, n] = self.log_emissions[k, xs[n]] + prop_of_most_likely_old_z

        return omega


    def backtrack(self, omega, xs):
        N = len(omega[0])

        zs = np.empty(N, np.int)
        xs = observables_to_indexs(self.observables, xs)

        log_likelihood = np.max(omega[:, N - 1])
        zs[N-1] = np.argmax(omega[:, N - 1])

        for n in range(N - 2, -1, -1):
            zs[n] = np.argmax([self.log_emissions[zs[n+1], xs[n+1]] + self.log_transitions[z, zs[n+1]] + o for z, o in enumerate(omega[:, n])])

        decoding = "".join([self.hidden_to_annotations[z] for z in zs])
        return log_likelihood, decoding

