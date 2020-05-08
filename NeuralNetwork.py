class NeuralNetwork:
    def __init__(self):
        """Initialize neural network with an empty array of layers and empty variables for
        the loss and loss's derivative"""
        self.layers = []
        self.loss = None
        self.lossDerivative = None

    def add(self, layer):
        """Adds a layer to the neural network"""
        self.layers.append(layer)

    def setLoss(self, inputLoss, inputLossDerivative):
        """Sets loss to use"""
        self.loss = inputLoss
        self.lossDerivative = inputLossDerivative

    def predict(self, inputData):
        """Return predictions for a set of given data"""
        results = []

        # For each input, feed the input through each layer, then append the output of the neuralnetwork to results
        for input in range(len((inputData))):
            output = inputData[input]
            for layer in self.layers:
                output = layer.feedForward(output)
            results.append(output)

        return results

    def train(self, inputs, expected, epochs, learningRate):
        """Trains the neural network using a set of inputs and expected value"""
        # For each epoch
        for epoch in range(epochs):
            displayError = 0
            # For each input
            for input in range(len(inputs)):
                output = inputs[input]
                # Feed input through neural network
                for layer in self.layers:
                    output = layer.feedForward(output)

                # Calculate loss
                displayError += self.loss(expected[input], output)

                # Backprop
                error = self.lossDerivative(expected[input], output)
                for layer in reversed(self.layers):
                    # Adjust the weights and biases of current layer, then return inputError
                    # for backPropogation of previous layer
                    error = layer.backProp(error, learningRate)

            # Calculates average error, displays epoch and error
            displayError /= len(inputs)
            print("Epoch {}/{}  with error {}".format(epoch + 1, epochs, displayError))