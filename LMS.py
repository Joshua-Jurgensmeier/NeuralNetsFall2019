import numpy as np


class ADALINENetwork:
    def __init__(self, W, b):
        # Augment W with column vector b to get X
        self.X = np.c_[W, b]

    # Takes training data [(input vector, output vector), ...]
    # and updates weights/biases
    def LMS_train(self, training_set, alpha):
        for example in training_set:
            # Calculate error
            p = example[0]
            t = example[1]
            e = t - self.a(p)

            # Update weights
            p = np.c_[example[0], [[1]]]
            self.X = self.X + (2 * alpha * (e.T @ p))

    # Takes input vector and returns ouput vector a
    def a(self, p):
        p = np.c_[p, [[1]]]  # append bias "input"
        return self.X @ p.T


# Main
if __name__ == "__main__":
    training_set = [
        (np.array([[1, 1]]), np.array([[1]])),
        (np.array([[1, -1]]), np.array([[-1]])),
    ]

    # Initialize weight matrix W and bias vector b
    W = np.array([[0, 0]])
    b = np.array([[0]])
    ada = ADALINENetwork(W, b)

    print("Untrained")
    print(ada.X)
    for i in range(5):
        ada.LMS_train(training_set, 0.1)
        print("After {} trainings:".format(i+1))
        print(ada.X)
