from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Flatten[
    in_neurons: Int,
    out_neurons: Int,
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward(self, x: Tensor) raises -> Tensor:
        return x.flatten()
