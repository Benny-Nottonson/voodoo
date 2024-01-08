from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Conv2D[
    in_channels: Int,
    out_channels: Int,
    kernel_width: Int,
    kernel_height: Int,
    stride: Int,
    padding: Int,
    use_bias: Bool = False,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    # TODO: add activation, regularizer, constraint, add 2d strides, add filters
](BaseLayer):
    var kernels: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.kernels = (
            Tensor(shape(out_channels, in_channels, kernel_width, kernel_height))
            .initialize[weight_initializer, weight_mean, weight_std]()
            .requires_grad()
        )

        @parameter
        if self.use_bias:
            self.bias = (
                Tensor(shape(out_channels, 1, 1))
                .initialize[bias_initializer, bias_mean, bias_std]()
                .requires_grad()
            )
        else:
            self.bias = Tensor(shape(0))

    fn forward(self, x: Tensor) raises -> Tensor:
        let res = conv_2d(x, self.kernels, self.padding, self.stride)

        @parameter
        if self.use_bias:
            return res + self.bias

        return res
