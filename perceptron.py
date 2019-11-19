import numpy as np


class PerceptronNetwork:
    def __init__(self, W, b):
        # Augment W with column vector b to get Wb
        self.Wb = np.c_[W, b]

    def _hardlim(self, x):
        return x >= 0

    # takes single training point, and updates weight matrix/bias vector
    # returns False if no updates were made and True otherwise
    def _update(self, p, t):
        e = t.T - self.a(p)  # Calculate error for all neurons
        p = np.c_[p, [[1]]]  # append bias "input"
        self.Wb = self.Wb + (e @ p)  # Apply perceptron learning rule
        # Return False if e is all zeros
        return e.any()

    # Takes training data [(input vector, output vector), ...]
    # and updates weights/biases
    def train(self, training_set):
        untrained = True
        while(untrained):
            untrained = False
            print("Training...")
            for example in training_set:
                untrained = self._update(example[0], example[1]) or untrained

    # Takes input vector and returns network classification ouput vector a
    def a(self, p):
        p = np.c_[p, [[1]]]  # append bias "input"
        return self._hardlim(self.Wb @ p.T)


def E4_4():
    # Setup
    # ([[weight, ear length]], [[rabbit==0 | bear==1]])
    training_set = [
        (np.array([[-1, 1]]), np.array([[1]])),
        (np.array([[-1, -1]]), np.array([[1]])),
        (np.array([[0, 0]]), np.array([[0]])),
        (np.array([[1, 0]]), np.array([[0]])),
    ]

    # Initialize weight matrix W and bias vector b
    W = np.array([[0, 0]])
    b = np.array([[0]])
    ntwk = PerceptronNetwork(W, b)

    print("Untrained weight + bias matrix")
    print(ntwk.Wb)
    print()

    ntwk.train(training_set)

    print()
    print("Trained weight + bias matrix")
    print(ntwk.Wb)
    print()

    # ii

    testing_set = [
        np.array([[-2, 0]]),
        np.array([[1, 1]]),
        np.array([[0, 1]]),
        np.array([[-1, -2]]),
    ]

    # Display network output for all test input vectors
    for example in testing_set:
        print()
        print("p = ", example)
        print("a = ", ntwk.a(example))


def E4_11():
    # Setup
    # ([[weight, ear length]], [[rabbit==0 | bear==1]])
    training_set = [
        (np.array([[1, 4]]), np.array([[0]])),
        (np.array([[1, 5]]), np.array([[0]])),
        (np.array([[2, 4]]), np.array([[0]])),
        (np.array([[2, 5]]), np.array([[0]])),
        (np.array([[3, 1]]), np.array([[1]])),
        (np.array([[3, 2]]), np.array([[1]])),
        (np.array([[4, 1]]), np.array([[1]])),
        (np.array([[4, 2]]), np.array([[1]])),
    ]

    # i
    # Initialize weight matrix W and bias vector b
    W = np.array([[-0.3, 0.3]])
    b = np.array([[-1]])
    fuzzy = PerceptronNetwork(W, b)

    print("Untrained weight + bias matrix")
    print(fuzzy.Wb)
    print()

    fuzzy.train(training_set)

    print()
    print("Trained weight + bias matrix")
    print(fuzzy.Wb)
    print()

    # ii
    # Display network output for all example input vectors
    for example in training_set:
        print()
        print("p = ", example[0])
        print("a = ", fuzzy.a(example[0]))

    # iii

# Main
if __name__ == "__main__":
    E4_11()
