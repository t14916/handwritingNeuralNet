import numpy as np
import scipy.special

class neuralNetwork:

    def __init__(self, layers,  learning_rate = 0.5):
        # initializer for neuralNetwork class, requires learning rate and a variable number of layers, which
        # which are given by list layers for which the length of the list represents the # of layers
        # and each list represents the layer
        self.learning_rate = learning_rate
        # layers is a list of layers, with the first being the input layer and the final being the output layer
        # Each element in layers refers to the number of weights in said layer.
        # NOTE: cannot be a Numpy array (Those do not support jagged arrays natively
        self.layers = layers

        # to help create weight matrices easily
        layer_next = [(layer, self.layers[index + 1]) for index, layer in enumerate(self.layers[:len(self.layers) - 1])]

        # link weight matrix, w_i_j for weight from node i to node j in the next layer
        # column : element of the previous layer
        # row : element of the next layer
        # list of numpy matrices which contain weights, as described above
        # weights are determined randomly along a normal distribution around 0 and around
        self.weights = [np.random.normal(0.0, pow(l[0], -0.5), (l[1], l[0])) for l in layer_next]

        # sets the activation function
        # primarily for testing purposes(we can change this later)
        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self, input):
        # runs through the whole matrix and given an input, spits out an output given the current weight matrix
        # uses query_helper

        layers = self.query_helper(input)

        return layers[-1]

    def query_helper(self, input):
        # runs through the whole matrix and given an input, spits out an output given the current weight matrix
        # Also converts it into a numpy array
        prev_layer = np.array(input, ndmin = 2)

        layers = [input]
        for layer in self.weights:

            # multiplies previous layer with the weights to get the next layer
            next_layer = np.dot(layer, prev_layer)

            # applies activation function on the next layer
            next_layer = self.activation_function(next_layer)

            # appends the you
            layers.append(next_layer)
            # sets previous layer as next layer
            prev_layer = next_layer

        return layers

    def train(self, input_list, target_list):

        # need to calculate hidden layer output as well, redo :/

        # converts list into a numpy array
        targets_iterator = iter(target_list)

        for input in input_list:
            # finds output using code from query function

            layers = self.query_helper(input)

            output = layers[-1]

            # finds the appropriate target list and turns it into a numpy array
            target = np.array(next(targets_iterator), ndmin=2)

            # calculate the final error
            output_error = target - output

            # Back Propogation
            # Goes through layers in reverse, in order to back propogate
            # Creates a new numpy array which can be reversed again and then added to self.weights

            #index for layer iteration
            i = 0

            for layer in np.fliplr(self.weights):

                layer_error = np.dot(layer, output_error)
                self.learning_rate*np.dot(output)

