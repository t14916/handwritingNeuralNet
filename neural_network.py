import numpy as np
import scipy.special as sp
import matplotlib.pyplot as mp

class NeuralNetwork:

    """
    NOTE: This class, and this project as a whole draws heavily from the book Make Your Own Neural Network by Tariq
    Rashid, but implements a slightly more complicated neural network which supports more than three layers.
    """
    def __init__(self, layers,  learning_rate = 0.5):
        """"
        initializer for neuralNetwork class, requires learning rate and a variable number of layers, which
        which are given by list layers for which the length of the list represents the # of layers
        and each list represents the layer.
        """
        self.learning_rate = learning_rate
        # layers is a list of layers, with the first being the input layer and the final being the output layer
        # Each element in layers refers to the number of weights in said layer.
        # NOTE: cannot be a Numpy array (Those do not support jagged arrays natively
        self.layers = layers

        # Enumerate(**) creates an enumerated list of (index, val) tuples up to and including the second to
        # last value of the list (last value of layers doesn't matter because it is the output, no weight matrix)
        # Whole list is a list of tuples of (layer_length, nextLayer_length)
        # to help create weight matrices easily
        layer_next = [(layer, self.layers[index + 1]) for index, layer in enumerate(self.layers[:len(self.layers) - 1])]

        # link weight matrix, w_i_j for weight from node i to node j in the next layer
        # row : element of the next layer (i)
        # column : element of the current layer (j)
        # list of numpy matrices which contain weights, as described above
        # weights are determined randomly along a normal distribution around 0 w/ std dev
        # 1/sqrt(len_column)
        self.weights = [np.random.normal(0.0, pow(l[0], -0.5), (l[1], l[0])) for l in layer_next]

        # sets the activation function
        # primarily for testing purposes(we can change this later)
        self.activation_function = lambda x: sp.expit(x)

    def query(self, input):
        """
        Runs through the whole matrix and given an input, returns an output given the current weight matrix.
        Throws an illegal exception if the input is of the incorrect size.
        """

        # Checks input size
        if len(input) != self.layers[0]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(input), self.layers[0]))

        # converts input to numpy array
        prev_layer = np.array(input, ndmin=2).T

        # First list is the input
        layers = [input]
        for weight_layer in self.weights:

            # print(prev_layer)

            # multiplies previous layer with the weights to get the next layer
            next_layer = np.dot(weight_layer, prev_layer)

            # applies activation function on the next layer
            next_layer = self.activation_function(next_layer)

            # appends the next layer
            layers.append(next_layer)
            # sets previous layer as next layer
            prev_layer = next_layer

        # Last list in layers should be the output.
        return layers[-1]

    def train(self, input_list, target_list):
        """
        Runs through the whole matrix and given an input, and then compares output to target. Then compares output
        to target_list and backpropogates the error. Throws an illegal exception if the input or target is of the
        incorrect size.
        """
        # Checks input size
        if len(input_list) != self.layers[0]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(input_list), self.layers[0]))

        # Checks target size
        if len(target_list) != self.layers[-1]:
            raise Exception('Input has incorrect size. The size of input was {}, but should be {}.'
                            .format(len(target_list), self.layers[0]))

        # Converts target to numpy array
        targets = np.array(target_list, ndmin=2).T
        # Converts input to numpy array
        prev_layer = np.array(input_list, ndmin=2).T

        # converts target to target
        # First list is the input
        layers = [input_list]
        for weight_layer in self.weights:

            # multiplies previous layer with the weights to get the next layer
            next_layer = np.dot(weight_layer, prev_layer)

            # applies activation function on the next layer
            next_layer = self.activation_function(next_layer)

            # appends the next layer
            layers.append(next_layer)
            # sets previous layer as next layer
            prev_layer = next_layer

        # Last list in layers should be the output.
        output = layers[-1]

        # Finds the Output error
        output_error = targets - output

        # Backpropogation loop
        for i in range(1, len(self.weights)):

            # if statement sets values for the error of layer
            if i == 1:
                layer_error = output_error
            else:
                layer_error = np.dot(self.weights[-i], output_error)

            # Applies backpropogation function to weights matrix
            self.weights[-i] += self.learning_rate * np.dot((layer_error * layers[-i] * (1 - layers[-i])),
                                                            layers[-i - 1].T)


def showData():
    # Opens file of training data and compiles a list of all the training dat
    data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    # Splits the long string using commas as the delimiter
    all_values = data_list[0].split(',')

    # Turns the list of values into a numpy array in the shape of the original image
    # Note: asfarray turns strings into numbers
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))

    # Draws the above image array as a
    mp.imshow(image_array, cmap = 'Greys', interpolation = 'None')

def trainMNISTDataset():
    # Opens file of training data and compiles a list of all the training dat
    data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    # For loop runs through each set of training data
    for string_data in data_list:

        # Splits the long string using commas as the delimiter
        numerical_data = string_data.split(',')

        # Scale the input such that it is between 0.01 and 1 (not 0 to avoid 0 value inputs which can be problematic
        # Note: the data is in the range 0-255 because it is based of pixel data
        # Note: the first value of the numerical data is not needed because it indicates the correct output value
        scaled_input = (np.asfarray(numerical_data[1:]) / 255.0 * 0.99) + 0.01

        #print(scaled_input)

        # Generate the correct output, the target vector
        # Ten possible choices for digits (0-9)
        target_vec = np.zeros(10) + 0.01
        target_vec[int(numerical_data[0])] = 0.99
