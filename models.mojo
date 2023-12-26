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
struct Model[name: String, input_dim: Int, output_dim: Int, hiddenLayers: HiddenLayers, dropout: Dropout, loss: LossFunc, optimizer: Optimizer]:
    ...

@value
struct BasicCNN[name: String, input_dim: Int, output_dim: Int]:
    var layer: ReLU
    var dropout: Dropout
    var loss: MSELoss
    var optimizer: SGD

    fn __init__(inout self):
        self.layer = ReLU()
        self.dropout = Dropout(0.5)
        self.loss = MSELoss()
        self.optimizer = SGD(0.01)

    fn forward(inout self, owned x: Matrix) -> Matrix:
        x = self.layer(x)
        x = self.dropout(x)
        return x

    fn backward(inout self, owned x: Matrix) -> Matrix:
        x = self.dropout.backward(x)
        x = self.layer.derivative(x)
        return x

    fn update(inout self, owned x: Matrix) -> Matrix:
        self.optimizer.step()
        return x