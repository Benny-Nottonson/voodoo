from .tensor import Tensor, conv_2d
from .activations import *
from .initializers import *
from .utils.shape import shape


struct Dense[
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_initial: Float32 = 0.0,
    bias_initial: Float32 = 0.0,
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
        self.W = Tensor(shape(in_neurons, out_neurons))
        self.bias = Tensor(shape(out_neurons))
        self._initialize_weight()
        if self.use_bias:
            self._initialize_bias()

    fn forward(self, x: Tensor) raises -> Tensor:
        return self._forward(x @ self.W + (self.bias * Float32(self.use_bias)))

    fn _initialize_weight(inout self) raises:
        @parameter
        if self.weight_initializer == "glorot_normal":
            self.W = glorot_normal(self.W)
        elif self.weight_initializer == "glorot_uniform":
            self.W = glorot_uniform(self.W)
        elif self.weight_initializer == "he_normal":
            self.W = he_normal(self.W)
        elif self.weight_initializer == "he_uniform":
            self.W = he_uniform(self.W)
        elif self.weight_initializer == "identity":
            self.W = identity(self.W)
        elif self.weight_initializer == "lecun_normal":
            self.W = lecun_normal(self.W)
        elif self.weight_initializer == "lecun_uniform":
            self.W = lecun_uniform(self.W)
        elif self.weight_initializer == "ones":
            self.W = ones(self.W)
        elif self.weight_initializer == "random_normal":
            self.W = random_normal(self.W)
        elif self.weight_initializer == "random_uniform":
            self.W = random_uniform(
                self.W, self.uniform_weight_min, self.uniform_weight_max
            )
        elif self.weight_initializer == "truncated_normal":
            self.W = truncated_normal(self.W)
        elif self.weight_initializer == "zeros":
            self.W = zeros(self.W)
        elif self.weight_initializer == "constant":
            self.W = constant(self.W, self.weight_initial)
        else:
            raise Error("Invalid weight initializer: " + self.weight_initializer)

    fn _initialize_bias(inout self) raises:
        @parameter
        if self.bias_initializer == "glorot_normal":
            self.bias = glorot_normal(self.bias)
        elif self.bias_initializer == "glorot_uniform":
            self.bias = glorot_uniform(self.bias)
        elif self.bias_initializer == "he_normal":
            self.bias = he_normal(self.bias)
        elif self.bias_initializer == "he_uniform":
            self.bias = he_uniform(self.bias)
        elif self.bias_initializer == "identity":
            self.bias = identity(self.bias)
        elif self.bias_initializer == "lecun_normal":
            self.bias = lecun_normal(self.bias)
        elif self.bias_initializer == "lecun_uniform":
            self.bias = lecun_uniform(self.bias)
        elif self.bias_initializer == "ones":
            self.bias = ones(self.bias)
        elif self.bias_initializer == "random_normal":
            self.bias = random_normal(self.bias)
        elif self.bias_initializer == "random_uniform":
            self.bias = random_uniform(
                self.bias, self.uniform_bias_min, self.uniform_bias_max
            )
        elif self.bias_initializer == "truncated_normal":
            self.bias = truncated_normal(self.bias)
        elif self.bias_initializer == "zeros":
            self.bias = zeros(self.bias)
        elif self.bias_initializer == "constant":
            self.bias = constant(self.bias, self.bias_initial)
        else:
            raise Error("Invalid bias initializer: " + self.bias_initializer)

    fn _forward(self, x: Tensor) raises -> Tensor:
        @parameter
        if self.activation == "elu":
            return elu(x)
        elif self.activation == "exp":
            return exp(x)
        elif self.activation == "gelu":
            return gelu(x)
        elif self.activation == "h_sig":
            return h_sig(x)
        elif self.activation == "linear":
            return linear(x)
        elif self.activation == "mish":
            return mish(x)
        elif self.activation == "relu":
            return relu(x)
        elif self.activation == "selu":
            return selu(x)
        elif self.activation == "sig":
            return sig(x)
        elif self.activation == "softmax":
            return softmax(x)
        elif self.activation == "softplus":
            return softplus(x)
        elif self.activation == "softsign":
            return softsign(x)
        elif self.activation == "swish":
            return swish(x)
        elif self.activation == "tanh":
            return tanh(x)
        elif self.activation == "none":
            return x
        else:
            raise Error("Invalid activation: " + self.activation)


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
