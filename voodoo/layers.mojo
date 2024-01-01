from .tensor import Tensor, conv_2d
from .activations import get_activation_code
from .utils.shape import shape

# TODO: There has to be a better way for this
# TODO: UseBias default changes depending on layer, see above
struct Layer[
    type: String,
    # Dense Parameters
    activation: String = "none",
    use_bias: Bool = True,
    weight_initializer: String = "he_normal",
    bias_initializer: String = "he_normal",
    weight_mean: Float32 = 0.0,
    weight_std: Float32 = 0.05,
    bias_mean: Float32 = 0.0,
    bias_std: Float32 = 0.05,
    # TODO: Add regularizers, constraints
    # Conv2d Parameters
    padding: Int = 0,
    stride: Int = 1,
    kernel_width: Int = 3,
    kernel_height: Int = 3,
    # LeakyRelu Parameters
    alpha: Float32 = 0.3,
]:
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
        in_neurons: Int,
        out_neurons: Int,
    ) raises:
        self.W = self.bias = Tensor(shape(0))
        # TODO: Extract into a dict once supported, since some have the same logic (Dense, LeakyRelu, etc.)
        @parameter
        if type == "dense":
            self.init_dense(in_neurons, out_neurons)
        elif type == "conv2d":
            self.init_conv2d(in_neurons, out_neurons)
        elif type == "leaky_relu":
            self.init_leaky_relu(in_neurons, out_neurons)
        else:
            raise Error("Invalid layer type: " + type)

    fn forward(self, x: Tensor) raises -> Tensor:
        @parameter
        if type == "dense":
            return self.forward_dense(x)
        elif type == "conv2d":
            return self.forward_conv2d(x)
        elif type == "leaky_relu":
            return self.forward_leaky_relu(x)
        else:
            raise Error("Invalid layer type: " + type)
            
    # Dense
    fn init_dense(
        inout self,
        in_neurons: Int,
        out_neurons: Int,
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).initialize[weight_initializer, weight_mean, weight_std]()
        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_neurons)).initialize[bias_initializer, bias_mean, bias_std]()
        else:
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()
    
    fn forward_dense(self, x: Tensor) raises -> Tensor:
        @parameter
        if self.activation == "none":
            return x @ self.W + (self.bias * Float32(self.use_bias))
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[get_activation_code[activation]()]()

    # TODO: Test
    # Conv2d
    fn init_conv2d(
        inout self,
        in_channels: Int,
        out_channels: Int,
    ) raises:
        self.W = Tensor(shape(out_channels, in_channels, kernel_width, kernel_height)).initialize[weight_initializer, weight_mean, weight_std]().requires_grad()
        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_channels, 1, 1)).initialize[bias_initializer, bias_mean, bias_std]()
        else:
            self.bias = Tensor(shape(out_channels, 1, 1)).initialize["zeros", 0.0]()

    fn forward_conv2d(self, x: Tensor) raises -> Tensor:
        let res = conv_2d(x, self.W, self.padding, self.stride)
        @parameter
        if self.use_bias:
            return res + self.bias
        return res

    # Dense
    fn init_leaky_relu(
        inout self,
        in_neurons: Int,
        out_neurons: Int,
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).initialize[weight_initializer, weight_mean, weight_std]()
        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_neurons)).initialize[bias_initializer, bias_mean, bias_std]()
        else:
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()
    
    fn forward_leaky_relu(self, x: Tensor) raises -> Tensor:
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[lrelu_code, self.alpha]()