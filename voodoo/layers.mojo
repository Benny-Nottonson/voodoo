from .tensor import Tensor, conv_2d
from .activations import get_activation_code
from .utils.shape import shape

struct Dense[
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_initial: Float32 = 0.0,
    weight_initial_std: Float32 = 0.05,
    bias_initial: Float32 = 0.0,
    bias_initial_std: Float32 = 0.05,
    uniform_weight_min: Float32 = -0.05,
    uniform_weight_max: Float32 = 0.05,
    uniform_bias_min: Float32 = -0.05,
    uniform_bias_max: Float32 = 0.05,
    # TODO: Add regularizers, constraints
]:
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
        in_neurons: Int,
        out_neurons: Int,
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).initialize[weight_initializer, weight_initial, weight_initial_std]()
        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_neurons)).initialize[bias_initializer, bias_initial, bias_initial_std]()
        else:
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()

    fn forward(self, x: Tensor) raises -> Tensor:
        @parameter
        if self.activation == "none":
            return x @ self.W + (self.bias * Float32(self.use_bias))
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[get_activation_code[activation]()]()

"""
struct Conv2d[
    padding: Int,
    stride: Int,
    use_bias: Bool = False,
]:
    var kernels: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_width: Int,
        kernel_height: Int,
    ) raises:
        self.kernels = (
            Tensor(shape(out_channels, in_channels, kernel_width, kernel_height))
            .randhe()
            .requires_grad()
        )
        self.bias = Tensor(shape(out_channels, 1, 1)).randhe().requires_grad()

    fn forward(self, x: Tensor) raises -> Tensor:
        let res = conv_2d(x, self.kernels, self.padding, self.stride)
        if self.use_bias:
            return res + self.bias
        return res
"""
