from voodoo import Tensor
from voodoo.initializers import NoneInitializer
from voodoo.constraints import NoneConstraint


trait BaseLayer:
    fn __init__(inout self) raises:
        ...

    fn forward(
        self, x: Tensor
    ) raises -> Tensor[x.shape, NoneInitializer, NoneConstraint, False, False]:
        ...
