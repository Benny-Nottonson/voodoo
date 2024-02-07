from voodoo import Tensor, Vector
from tensor import TensorShape
from voodoo.initializers import NoneInitializer
from voodoo.constraints import NoneConstraint


struct Reshape[new_shape: TensorShape]():
    fn __init__(inout self) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[new_shape, NoneInitializer, NoneConstraint, False, False]:
        return x.reshape[new_shape]()
