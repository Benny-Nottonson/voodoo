# from voodoo import Tensor
# from .BaseLayer import BaseLayer


# struct MaxPool2D[
#     kernel_width: Int,
#     kernel_height: Int,
#     stride: Int = 1,
#     padding: Int = 0,
# ](BaseLayer):
#     fn __init__(
#         inout self,
#     ) raises:
#         ...

#     fn forward(self, x: Tensor) raises -> Tensor[False, False]:
#         let res = x.maxpool_2d(
#             StaticIntTuple[2](kernel_width, kernel_height),
#             stride,
#             padding,
#         )
#         return res
