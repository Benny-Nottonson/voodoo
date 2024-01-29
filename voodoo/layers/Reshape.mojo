from voodoo import Tensor, Vector
from tensor import TensorShape
from .BaseLayer import BaseLayer


struct Reshape[new_shape: TensorShape](BaseLayer):
    var _new_shape: Vector[Int]

    fn __init__(inout self) raises:
        self._new_shape = Vector[Int](new_shape.rank())
        for i in range(new_shape.rank()):
            self._new_shape.store(i, new_shape[i])

    @always_inline("nodebug")
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.reshape(self._new_shape)
