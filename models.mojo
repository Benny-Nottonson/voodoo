from layers import Layer
from optimizers import Optimizer
from losses import LossFunction
from utilities import Matrix
from collections.vector import InlinedFixedVector

trait Model:
    fn fit(inout self, x: Matrix, y: Matrix, epochs: Int, batch_size: Int):
        ...

    fn predict(self, x: Matrix) -> Matrix:
        ...
    
struct Sequential(Model):
    var layers: InlinedFixedVector[Layer]
    var loss: LossFunction
    var optimizer: Optimizer

    fn add(inout self, layer: Layer):
        ...

    fn pop(inout self):
        ...

    fn fit(inout self, x: Matrix, y: Matrix, epochs: Int, batch_size: Int):
        ...

    fn predict(self, x: Matrix) -> Matrix:
        return Matrix(0, 0)