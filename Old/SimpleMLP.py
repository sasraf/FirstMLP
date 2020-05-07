import numpy as np

def sigmoid(input):
    """Sigmoid function where input is a numpy array or vector"""
    return 1.0 / (1.0 + np.exp(-input))

class SimpleMLP:

    def __init__ (self):
        """Initialize a neural network with 4 layers of size 784, 20, 20, and 10"""
        self.numberOfLayers = 4
        self.layerSizes = [784, 20, 20, 10]

        # Make an array to hold biases
        self.biases = []

        # For each set of biases n, make layerSizes[n + 1] number of biases (each neuron to the right of the biases has a bias)
        # Biases are of form biases[set][neuron]
        for set in (self.numberOfLayers - 1):
            self.biases = np.random.normal(size=(self.layerSizes[set + 1]))



        # Makes a list of random weights for each connection between neurons
        # self.weights = [np.random.rand(y, x) for x, y in zip(layerSizes[:-1], layerSizes[1:])]

        # Makes an array of arrays to hold weights
        self.weights = [[] for i in (self.numberOfLayers - 1)]

        # For each set of weights (there being a x - 1 set of weights for a MLP of size x)
        # Weights are of form weights[set][right neuron][left neurons]
        for set in (self.numberOfLayers - 1):
            # For each number of neurons that the weights of set n feed into (aka the layer is to the right of the weights)
            for neuron in self.layerSizes[set + 1]:
                # Randomly assign y weights for each set of weights for each neuron where y is the number of neurons that feed into the current neuron
                self.weights[set][neuron] = np.random.normal(size=(self.layerSizes[set]))

        # self.weights[0] = np.random.normal(size=(self.layerSizes[0] * self.layerSizes))


        # Creates random weights
        # for n in range(0, self.numberOfLayers - 1):
        #     self.weights[n] = np.random.normal(size=(self.layerSizes[n] * self.layerSizes[n + 1]))
            # for i in range(0, self.layerSizes[n] * self.layerSizes[n + 1]):
                # self.weights[n][i] = np.random


    def feedForward(self, inputLayer):
        """Returns the mlp's output for a given input"""

        # TODO: make more concise

        # Initialize hiddenLayerOne as an array of size 20
        hiddenLayerOne = [0] * self.layerSizes[1]

        # Calculates the sum of wx - b for all neurons in layerone then applies sigmoid function to all the neurons
        for neuron in range(len(hiddenLayerOne)):
            for x in range(len(inputLayer)):
                hiddenLayerOne[neuron] += self.weights[1][neuron][x] * inputLayer[x] - self.biases[1][neuron]
        hiddenLayerOne = sigmoid(hiddenLayerOne)

        # Initialize hiddenLayerTwo as an array of size 20
        hiddenLayerTwo = [0] * self.layerSizes[2]

        # Calculates the sum of wx - b for all neurons in layertwo then applies sigmoid function to all the neurons
        for neuron in range(len(hiddenLayerTwo)):
            for x in range(len(hiddenLayerOne)):
                hiddenLayerTwo[neuron] += self.weights[2][neuron][x] * hiddenLayerOne[x] - self.biases[2][neuron]
        hiddenLayerTwo = sigmoid(hiddenLayerTwo)

        outputLayer = [0] * self.layerSizes[3]
        for neuron in range(len(outputLayer)):
            for x in range(len(hiddenLayerTwo)):
                outputLayer[neuron] += self.weights[3][neuron][x] * hiddenLayerTwo[x] - self.biases[3][neuron]
        outputLayer = sigmoid(outputLayer)

        return outputLayer



        # int count = 0
        # for n in range(len(hiddenLayerOne)):
        #     for i in range(len(inputLayer)):
        #         # in self.weights[1][count], 1 refers to the first set of weights, and count
        #         hiddenLayerOne[n] += self.weights[1][count] - self.biases[1]
        #         count += 1
        # hiddenLayerOne = sigmoid(hiddenLayerOne)




    # def feedForward(self, input):
    #     """Returns the mlp's output for a given input"""
    #     for bias, weight in zip(self.biases, self.weights):
    #         input = sigmoid(np.dot(weight, input) + bias)
    #     return input