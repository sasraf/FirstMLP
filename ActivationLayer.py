from Layer import  Layer

class ActivationLayer(Layer):
    def __init__(self, passedActivationFunction, passedActivationDerivative):
        """Initializes activation function and derivative of activation function as parameters passed in"""
        self.activationFunction = passedActivationFunction
        self.activationDerivative = passedActivationDerivative

    def feedForward(self, passedInput):
        """Stores the passed input as input and returns the input passed through the activation function as output"""
        self.input = passedInput
        self.output = self.activationFunction(self.input)
        return self.output

    def backProp(self, outputError, learningRate):
        """Returns input error for a given output error. The input error of the activation function
        is equal to the output error * the derivative of the activation function at the input"""
        return self.activationDerivative(self.input) * outputError
