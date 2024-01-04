from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Conv2D[
    in_batches: Int,
    in_channels: Int,
    in_height: Int,
    in_width: Int,
    padding: Int = 0,
    stride: Int = 1,
    kernel_size: Int = 3,
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    # TODO: add activation, regularizer, constraint, add 2d strides, add filters
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = Tensor(
            shape(
                self.in_batches,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        ).initialize[weight_initializer, weight_mean, weight_std]()

        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(
            self.in_batches,
            self.in_channels,
            (self.in_width - kernel_size + 2 * padding) // stride + 1,
            (self.in_height - kernel_size + 2 * padding) // stride + 1,
        )).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
        else:
            self.bias = Tensor(shape(
            self.in_batches,
            self.in_channels,
            (self.in_width - kernel_size + 2 * padding) // stride + 1,
            (self.in_height - kernel_size + 2 * padding) // stride + 1,
        )).initialize["zeros", 0.0]()

    fn forward(self, x: Tensor) raises -> Tensor:
        let res = conv_2d(
            x,
            self.W,
            self.stride,
            self.padding,
        )

        @parameter
        if self.use_bias:
            return res + self.bias

        return res