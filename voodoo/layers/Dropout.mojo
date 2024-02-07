from voodoo import Tensor
from voodoo.initializers import NoneInitializer
from voodoo.constraints import NoneConstraint
from tensor import TensorShape


struct Dropout[
    dropout_rate: Float32 = 0.5, noise_shape: TensorShape = TensorShape(0)
]():
    fn __init__(
        inout self,
    ) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[x.shape, NoneInitializer, NoneConstraint, False, False]:
        return x.dropout[dropout_rate, noise_shape]()
