from voodoo import Tensor
from tensor import TensorShape
from .BaseLayer import BaseLayer


struct Conv2D[
    in_channels: Int,
    kernel_width: Int,
    kernel_height: Int,
    stride: Int,
    padding: Int,
    activation: String = "none",
    use_bias: Bool = False,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "zeros",
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
            TensorShape(in_channels, kernel_width, kernel_height)
        ).initialize[weight_initializer, weight_mean, weight_std]()
        self.W = self.W.requires_grad()

        @parameter
        if self.use_bias:
            self.bias = Tensor(TensorShape(in_channels, 1, 1)).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
            self.bias = self.bias.requires_grad()
        else:
            self.bias = Tensor(TensorShape(0))

    fn forward(self, x: Tensor) raises -> Tensor[False, False]:
        let res = x.conv_2d(self.W, self.padding, self.stride).compute_activation[
            self.activation
        ]()

        @parameter
        if self.use_bias:
            return res + self.bias

        return res
