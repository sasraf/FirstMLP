import numpy as np

from NeuralNetwork import  NeuralNetwork
from FullyConnectedLayer import FullyConnectedLayer
from ActivationLayer import ActivationLayer
from ActivationFunctions import tanh, tanhDerivative
from LossFunction import meanSquaredError, meanSquaredErrorDerivative

# Sample training data
inputData = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
expectedOutput = np.array([[[0]], [[1]], [[1]], [[0]]])

# Creating a neural network with 3 nodes in first hidden layer, 1 node in final layer, with activation functions
# after each layer
network = NeuralNetwork()
network.add(FullyConnectedLayer(2, 3))
network.add(ActivationLayer(tanh, tanhDerivative))
network.add(FullyConnectedLayer(3, 1))
network.add(ActivationLayer(tanh, tanhDerivative))

# Training network
network.setLoss(meanSquaredError, meanSquaredErrorDerivative)
network.train(inputData, expectedOutput, epochs=1000, learningRate=.1)

# Test the network
output = network.predict(inputData)
for set in range(len(inputData)):
    print("For set {} my prediction is {}. The correct value is {}".format(inputData[set], output[set], expectedOutput[set]))