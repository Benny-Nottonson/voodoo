from activation import *
from initializer import *
from layer import *
from loss import *
from optimizer import *
from regularizer import *

trait Model:
    ...

struct StaticTestModel[T: DType](Model):
    var name: String
    var inputSize: Int
    var outputSize: Int
    var hiddenLayers: StaticDense[T]
    var outputLayer: StaticDense[T]
    var loss: MeanSquaredError

    fn __init__(inout self, name: String, inputSize: Int, outputSize: Int, hiddenLayers: StaticDense[T], outputLayer: StaticDense[T], loss: MeanSquaredError):
        self.name = name
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenLayers = hiddenLayers
        self.outputLayer = outputLayer
        self.loss = loss

    fn forward(self, owned x: Tensor[T]) raises -> Tensor[T]:
        x = self.hiddenLayers.forward(x)
        x = self.outputLayer.forward(x)
        return x

    fn backward(inout self, owned x: Tensor[T], owned y: Tensor[T]) raises -> Tensor[T]:
        x = self.forward(x)
        let loss = self.loss.calculate(x, y)
        var grad = loss.delta
        grad = self.outputLayer.backward(x, grad)
        grad = self.hiddenLayers.backward(x, grad)
        return grad

    fn predict(self, owned x: Tensor[T]) raises -> Tensor[T]:
        return self.forward(x)

"""
struct Regression(Model):
    ...

struct Classifcation(Model):
    ...
"""