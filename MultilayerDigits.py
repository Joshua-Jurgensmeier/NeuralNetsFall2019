import numpy as np
import math
import idx2numpy
np.set_printoptions(linewidth=300)


class NeuralLayer:
    def __init__(self, f, df, num_ns, num_inputs):
        X_INIT_RANGE = 0.5
        self.num_ns = num_ns  # Number of Neurons in layer
        self.f = getattr(self, f)       # Transfer function of layer
        self.df = getattr(self, df)     # Derivative of transfer function
        self.a = None    # The current layer output computed by propagate
        self.s = None    # The current layer sensitivity computed by backpropagate
        self.dFn = None  # Diagonal matrix of df evaluated at n (For computing Jacobian)
        # The weight matrix augmented with bias vector, initialized to small random numbers
        self.X = np.random.uniform(
                -X_INIT_RANGE,
                X_INIT_RANGE,
                (num_ns, num_inputs+1)
            )

    def propagate(self, p):
        n = self.compute_n(p)
        # Update our a and dFn. MUST COMPUTE a FIRST!
        self.a = self.f(n)
        self.dFn = np.diagflat(self.df(n))

    def compute_n(self, p):
        # append bias "input"
        p = np.r_[p, [[1]]]
        # Apply weights/bias
        return self.X @ p

    def update_X(self, alpha):
        # Update X with iterative, stochastic gradient descent
        # Appends 1 to a.T to update bias
        self.X = self.X - alpha * self.s @ np.r_[self.prev_layer.a, [[1]]].T

    def update_s(self):
        # Only use weights from X to compute sensitivity, not biases
        self.s = self.dFn @ np.delete(self.next_layer.X, -1, 1).T @ self.next_layer.s

    def print_net(self):
        print("Output:\n{}".format(self.a))
        print("Weights:")
        print(self.num_ns)
        print(self.X)
        print()

    # Define transfer functions and their derivatives
    # The one the layer uses will be referenced by f
    def logsig(self, n):
        return 1/(1+np.exp(-n))

    # Uses a computed by logsig
    def dlogsig(self, n):
        return (1-self.a)*self.a

    # Computation simplified by multiplying by e**-max(n)/e**-max(n)
    def softmax(self, n):
        expn = np.exp(n - np.max(n))
        return expn / expn.sum(axis=0)

    # Uses a computed by softmax
    def dsoftmax(self, n):
        return (1-self.a)*self.a


class InputNeuralLayer(NeuralLayer):
    # If first layer we need to know the number of inputs explicitly
    def __init__(self, num_ns, f, df, num_inputs, next_layer):
        self.next_layer = next_layer
        super().__init__(f, df, num_ns, num_inputs)

    # Propagates forward network inputs
    def propagate(self, p):
        # Make sure we have a column vector
        if p.shape[0] == 1:
            p = p.T
        # Store input, to be used in backpropagation
        self.p = p
        # Compute and update
        super().propagate(p)
        # Update next layer
        self.next_layer.propagate()

    # Propagates backward network "sensitivities" and updates weights
    def backpropagate(self, alpha):
        self.update_s()
        self.X = self.X - alpha * self.s @ np.r_[self.p, [[1]]].T

    def compute_a(self, p):
        # Make sure we have a column vector
        if p.shape[0] == 1:
            p = p.T
        self.a = self.f(self.compute_n(p))
        self.next_layer.compute_a()

    def print_net(self):
        super().print_net()
        self.next_layer.print_net()


class HiddenNeuralLayer(NeuralLayer):
    def __init__(self, num_ns, f, df, prev_layer, next_layer):
        self.prev_layer = prev_layer
        self.next_layer = next_layer
        super().__init__(f, df, num_ns, self.prev_layer.num_ns)

    # Propagates forward network inputs
    def propagate(self):
        # Update our a
        super().propagate(self.prev_layer.a)
        # Update next layer
        self.next_layer.propagate()

    # Backpropagates network "sensitivities" and updates weights
    def backpropagate(self, alpha):
        self.update_s()
        self.update_X(alpha)

        self.prev_layer.backpropagate(alpha)

    def compute_a(self):
        self.a = self.f(self.compute_n(self.prev_layer.a))
        self.next_layer.compute_a()

    def print_net(self):
        super().print_net()
        self.next_layer.print_net()


class OutputNeuralLayer(NeuralLayer):
    def __init__(self, num_ns, f, df, prev_layer):
        self.prev_layer = prev_layer
        super().__init__(f, df, num_ns, self.prev_layer.num_ns)

    # Propagates forward network inputs
    def propagate(self):
        # Update our a
        super().propagate(self.prev_layer.a)

    # Backpropagates network "sensitivities" and updates weights
    def backpropagate(self, alpha, t):
        # Make sure we have a column vector
        if t.shape[0] == 1:
            t = t.T
        # Compute sensitivity column vector
        self.s = -2 * self.dFn @ (t-self.a)
        self.update_X(alpha)

        self.prev_layer.backpropagate(alpha)

    def compute_a(self):
        self.a = self.f(self.compute_n(self.prev_layer.a))

    def print_net(self):
        super().print_net()


# Multilayer nueral network as doubly linked list of layers
class NeuralNetwork:
    def __init__(self, num_inputs, layers_args):
        # Instantiate and link up all layers
        input_args = layers_args.pop(0)
        output_args = layers_args.pop(-1)

        # Instantiate input layer
        self.input_layer = InputNeuralLayer(
            input_args['num_ns'],
            input_args['f'],
            input_args['df'],
            num_inputs,
            None
        )

        # Instantiate and link up hidden layers
        prev_layer = self.input_layer
        for layer_args in layers_args:
            prev_layer.next_layer = HiddenNeuralLayer(
                layer_args['num_ns'],
                layer_args['f'],
                layer_args['df'],
                prev_layer,
                None
            )
            prev_layer = prev_layer.next_layer

        # Instantiate and link up hidden layer
        self.output_layer = OutputNeuralLayer(
            output_args['num_ns'],
            output_args['f'],
            output_args['df'],
            prev_layer
        )
        prev_layer.next_layer = self.output_layer      

    # Do iterative backpropagation Least-mean-square stochastic gradient descent
    def train(self, trainingExamples, trainingLabels, alpha):
        for example, label in zip(trainingExamples, trainingLabels):
            self.input_layer.propagate(example)
            self.output_layer.backpropagate(alpha, label)

    def classify(self, example):
        self.input_layer.compute_a(example)
        return self.output_layer.a.argmax()

    # Compute the percent classified correctly
    def test(self, testExamples, testLabels):
        total = testLabels.shape[0]
        correct = 0
        for example, label in zip(testExamples, testLabels):
            if self.classify(example) == label:
                correct += 1
        return correct / total

    def print_net(self):
        self.input_layer.print_net()


# Functions to prepare MNIST data for input in my NN
# Normalize images and flatten into row vectors for input into NN 
def parse_idx_images(idx_images):
    image_size = idx_images.shape[1] * idx_images.shape[2]
    idx_images = idx_images / 255
    flat = idx_images.reshape((idx_images.shape[0], image_size))
    return np.expand_dims(flat, 1)


# Encodes scalar as vector with a 1 at the corresponding index
def parse_idx_labels(idx_labels, categories):
    vector_labels = np.zeros((idx_labels.shape[0], categories))
    for i in range(idx_labels.shape[0]):
        scalar_label = idx_labels[i]
        vector_labels[i][scalar_label] = 1
    return np.expand_dims(vector_labels, 1)


# Main
if __name__ == "__main__":

    ALPHA = 0.2

    h = {
        'num_ns': 150,
        'f': 'logsig',
        'df': 'dlogsig',
    }
    o = {
        'num_ns': 10,
        'f': 'softmax',
        'df': 'dsoftmax',
    }

    nn = NeuralNetwork(784, [h, o])

    # Training data
    train_examples = idx2numpy.convert_from_file("MNIST_digits\\train-images-idx3-ubyte")
    s_train_labels = idx2numpy.convert_from_file("MNIST_digits\\train-labels-idx1-ubyte")

    # Testing data
    test_examples = idx2numpy.convert_from_file("MNIST_digits\\t10k-images-idx3-ubyte")
    test_labels = idx2numpy.convert_from_file("MNIST_digits\\t10k-labels-idx1-ubyte")

    # Get inputs in right dimensions and the such
    flat_train_examples = parse_idx_images(train_examples)
    v_train_labels = parse_idx_labels(s_train_labels, 10)

    flat_test_examples = parse_idx_images(test_examples)
    #v_test_labels = parse_idx_labels(test_labels, 10)

    # Train
    nn.train(flat_train_examples, v_train_labels, ALPHA)
    

    print(test_examples[49])
    print("Network Classification: ", nn.classify(flat_test_examples[49]))
    print("Label: ", test_labels[49])


    print(nn.test(flat_train_examples, s_train_labels))
    print(nn.test(flat_test_examples, test_labels))
