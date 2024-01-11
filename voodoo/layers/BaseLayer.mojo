from voodoo import Tensor


trait BaseLayer:
    fn __init__(inout self) raises:
        ...

    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        ...
