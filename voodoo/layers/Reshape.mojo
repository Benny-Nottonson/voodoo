from voodoo import Tensor, shape, Vector
from .BaseLayer import BaseLayer


struct Reshape[new_shape: DynamicVector[Int]](BaseLayer):
    var _new_shape: Vector[Int]

    fn __init__(inout self) raises:
        self._new_shape = Vector[Int](len(new_shape))
        for i in range(len(new_shape)):
            self._new_shape.store(i, new_shape[i])

    @always_inline
    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        return x.reshape(self._new_shape)
