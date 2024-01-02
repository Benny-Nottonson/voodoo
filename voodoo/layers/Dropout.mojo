from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Dropout[
    in_neurons: Int,
    out_neurons: Int,
    dropout_rate: Float32 = 0.5,
    noise_shape: DynamicVector[Int] = DynamicVector[Int](),
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.dropout[dropout_rate, noise_shape]()
