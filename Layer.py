# Abstract class

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Calculates output of layer for a given input
    def feedForward(self, passedInput):
        raise NotImplementedError

    # Calculates dE/dX for a dE/dY as well as any output parameters
    def backProp(self, outputError, learningRate):
        raise NotImplementedError
