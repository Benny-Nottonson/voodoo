from voodoo import Tensor, conv_2d, get_activation_code, shape
from .BaseLayer import BaseLayer


struct Conv2D[
    in_neurons: Int,
    out_neurons: Int,
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    padding: Int = 0,
    stride: Int = 1,
    kernel_width: Int = 3,
    kernel_height: Int = 3,
](BaseLayer):
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = Tensor(
            shape(
                self.out_neurons,
                self.in_neurons,
                self.kernel_width,
                self.kernel_height,
            )
        ).initialize[weight_initializer, weight_mean, weight_std]()

        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(self.out_neurons, 1, 1)).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
        else:
            self.bias = Tensor(shape(self.out_neurons, 1, 1)).initialize["zeros", 0.0]()

    fn forward(self, x: Tensor) raises -> Tensor:
        return conv_2d(
            x,
            self.W,
            self.stride,
            self.padding,
        ) + (self.bias * Float32(self.use_bias))
