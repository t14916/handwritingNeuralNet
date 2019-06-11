import numpy as np
import scipy.special as sp
import matplotlib.pyplot as mp


class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: sp.expit(x)
        self.inverse_activation_function = lambda x: sp.logit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        np.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = np.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = np.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs

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
        # NOTE: cannot be a np array (Those do not support jagged arrays natively
        self.layers = layers

        # Enumerate(**) creates an enumerated list of (index, val) tuples up to and including the second to
        # last value of the list (last value of layers doesn't matter because it is the output, no weight matrix)
        # Whole list is a list of tuples of (layer_length, nextLayer_length)
        # to help create weight matrices easily
        layer_next = [(layer, self.layers[index + 1]) for index, layer in enumerate(self.layers[:len(self.layers) - 1])]

        # link weight matrix, w_i_j for weight from node i to node j in the next layer
        # row : element of the next layer (i)
        # column : element of the current layer (j)
        # list of np matrices which contain weights, as described above
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

        # converts input to np array
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

        # Converts target to np array
        targets = np.array(target_list, ndmin=2).T
        # Converts input to np array
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

        # loop to create a list of numpy arrays of errors in reverse order (i.e. output error is first error)
        error_array = []
        for i in range(1, len(self.weights)):
            if i == 1:
                # first error is output error
                error_array += [output_error]
            else:
                error_array += [np.dot(self.weights[-i + 1].T, output_error)]

        # Backpropogation loop
        for i in range(1, len(self.weights)):
            # Applies backpropogation function to weights matrix
            self.weights[-i] += self.learning_rate * np.dot((error_array[i - 1] * layers[-i] * (1.0 - layers[-i])),
                                                            layers[-i - 1].T)

def showData():
    # Opens file of training data and compiles a list of all the training dat
    data_file = open("mnist_dataset/mnist_train.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    # Splits the long string using commas as the delimiter
    all_values = data_list[0].split(',')

    # Turns the list of values into a np array in the shape of the original image
    # Note: asfarray turns strings into numbers
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))

    # Draws the above image array as a
    mp.imshow(image_array, cmap = 'Greys', interpolation = 'None')


def trainAndTestMNISTDataset(epochs):
    # Opens file of training data and compiles a list of all the training dat
    data_file = open("mnist_dataset/mnist_train.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    # Initializes a neural network with 784 input nodes (for each of the pixels in the image) and 10 ending nodes
    # (one for each of the possibilities 0-9 for the handwritten letters)
    nn = NeuralNetwork([784, 100, 10], 0.2)
    count = 0
    # For loop runs through each set of training data
    for _ in range(epochs):
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

            # Train the Neural Network
            nn.train(scaled_input, target_vec)

    # print(n.wih)
    # print(nn.weights)

    # Load the test data
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for test_string_data in test_data_list:
        test_data = test_string_data.split(',')

        scaled_input = (np.asfarray(test_data[1:]) / 255.0 * 0.99) + 0.01
        correct_result = int(test_data[0])

        n_output = nn.query(scaled_input)

        max_index = n_output.argmax()

        if max_index == correct_result:
            count += 1

        # print(max_index == correct_result)
        # print(max_index)
        # print(correct_result)
        # print(n_output)

    print(count / 10000)


trainAndTestMNISTDataset(1)
#showData()
