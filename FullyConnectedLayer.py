from Layer import Layer
import numpy as np

# Inherits from Layer class
class FullyConnectedLayer(Layer):
    def __init__(self, inputSize, outputSize):

        # Creates weights as a randomized matrix of inputSize x outputSize
        self.weights = np.random.rand(inputSize, outputSize) -.5

        # Creates biases as a randomized 1 x outputSize matrix
        self.biases = np.random.randn(1, outputSize) -.5

    def feedForward(self, input):
        self.input = input

        # returns a 1 x outPutSize matrix of the output values corresponding to each neuron in the next layer
        self.output = np.dot(self.input, self.weights) + self.biases

        return  self.output

    def backProp(self, outputError, learningRate):
        # Input error is equal to the dotproduct of the output error and transpose of the weights
        inputError = np.dot(outputError, self.weights.T)

        # Weight error is equal to the dot product of the transpose of the inputs with the outputError
        weightsError = np.dot(self.input.T, outputError)

        # Update weights and biases
        self.weights -= learningRate * weightsError
        self.biases -= learningRate * outputError

        return inputError