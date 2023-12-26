from layers import *
from losses import *
from optimizers import *

trait GenericModel:
    fn forward(inout self, x: Matrix) -> Matrix:
        ...

    fn backward(inout self, x: Matrix) -> Matrix:
        ...

    fn update(inout self, x: Matrix) -> Matrix:
        ...

@value
struct Model[name: String, input_dim: Int, output_dim: Int, hiddenLayers: Int, dropout: Dropout, loss: LossFunc, optimizer: Optimizer]:
    ...

@value
struct BasicCNN[name: String, input_dim: Int, output_dim: Int]:
    var hiddenLayers: HiddenLayers[1]
    var dropout: Dropout
    var loss: MSELoss
    var optimizer: SGD

    fn __init__(inout self):
        self.hiddenLayers = HiddenLayers[1](DynamicVector[Int](1))
        self.dropout = Dropout(0.5)
        self.loss = MSELoss()
        self.optimizer = SGD(0.01)

    fn forward(inout self, owned x: Matrix) -> Matrix:
        x = self.hiddenLayers.forward(x)
        x = self.dropout.forward(x)
        return x

    fn backward(inout self, owned x: Matrix) -> Matrix:
        x = self.dropout.backward(x)
        x = self.hiddenLayers.backward(x)
        return x

    fn update(inout self, owned x: Matrix) -> Matrix:
        x = self.hiddenLayers.update(x)
        return x