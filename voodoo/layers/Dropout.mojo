# from voodoo import Tensor
# from .BaseLayer import BaseLayer


# struct Dropout[
#     dropout_rate: Float32 = 0.5,
#     noise_shape: DynamicVector[Int] = DynamicVector[Int](),
# ](BaseLayer):
#     fn __init__(
#         inout self,
#     ) raises:
#         ...

#
#     fn forward(self, x: Tensor) raises -> Tensor[False, False]:
#         return x.dropout[dropout_rate, noise_shape]()
