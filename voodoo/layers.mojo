from .tensor import Tensor, conv_2d
from .code_lookup import get_activation_code
from .utils.shape import shape


struct Layer[
    type: String,
    in_neurons: Int = 1,
    out_neurons: Int = 1,
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
    in_batches: Int = 1,
    in_channels: Int = 1,
    in_height: Int = 1,
    in_width: Int = 1,
    padding: Int = 0,
    stride: Int = 1,
    kernel_width: Int = 3,
    kernel_height: Int = 3,
    # LeakyRelu Parameters
    alpha: Float32 = 0.3,
    # Dropout Parameters
    dropout_rate: Float32 = 0.5,
    noise_shape: DynamicVector[Int] = DynamicVector[Int](),
    # Maxpool2d Parameters
    pool_size: Int = 2,
]:
    var W: Tensor
    var bias: Tensor

    fn __init__(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

        # TODO: Extract into a dict once supported, since some have the same logic (Dense, LeakyRelu, etc.)
        @parameter
        if type == "dense":
            self.init_dense()
        elif type == "conv2d":
            self.init_conv2d()
        elif type == "leaky_relu":
            self.init_leaky_relu()
        elif type == "dropout":
            self.init_dropout()
        elif type == "maxpool2d":
            self.init_maxpool2d()
        elif type == "flatten":
            self.init_flatten()
        else:
            raise "Invalid layer type: " + type

    fn forward(self, x: Tensor) raises -> Tensor:
        @parameter
        if type == "dense":
            return self.forward_dense(x)
        elif type == "conv2d":
            return self.forward_conv2d(x)
        elif type == "leaky_relu":
            return self.forward_leaky_relu(x)
        elif type == "dropout":
            return self.forward_dropout(x)
        elif type == "maxpool2d":
            return self.forward_maxpool2d(x)
        elif type == "flatten":
            return self.forward_flatten(x)
        else:
            raise "Invalid layer type: " + type

    # Dense
    fn init_dense(
        inout self,
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).initialize[
            weight_initializer, weight_mean, weight_std
        ]()

        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_neurons)).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
        else:
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()

    fn forward_dense(self, x: Tensor) raises -> Tensor:
        @parameter
        if self.activation == "none":
            return x @ self.W + (self.bias * Float32(self.use_bias))
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[
            get_activation_code[activation]()
        ]()

    # TODO: Test
    # Conv2d
    fn init_conv2d(
        inout self,
    ) raises:
        self.W = Tensor(
            shape(
                self.in_batches,
                self.in_channels,
                self.kernel_width,
                self.kernel_height,
            )
        ).initialize[weight_initializer, weight_mean, weight_std]()

        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(
            self.in_batches,
            self.in_channels,
            (self.in_width - kernel_width + 2 * padding) // stride + 1,
            (self.in_height - kernel_height + 2 * padding) // stride + 1,
        )).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
        else:
            self.bias = Tensor(shape(
            self.in_batches,
            self.in_channels,
            (self.in_width - kernel_width + 2 * padding) // stride + 1,
            (self.in_height - kernel_height + 2 * padding) // stride + 1,
        )).initialize["zeros", 0.0]()

    fn forward_conv2d(self, x: Tensor) raises -> Tensor:
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

    # TODO: Test
    # Maxpool2d
    fn init_maxpool2d(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward_maxpool2d(self, x: Tensor) raises -> Tensor:
        return x.max_pool_2d(self.pool_size, self.pool_size, self.stride, self.padding)

    # TODO: Test
    # Flatten
    fn init_flatten(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward_flatten(self, x: Tensor) raises -> Tensor:
        return x.flatten()

    # TODO: Test
    # Leaky Relu
    fn init_leaky_relu(
        inout self,
    ) raises:
        self.W = Tensor(shape(in_neurons, out_neurons)).initialize[
            weight_initializer, weight_mean, weight_std
        ]()

        @parameter
        if self.use_bias:
            self.bias = Tensor(shape(out_neurons)).initialize[
                bias_initializer, bias_mean, bias_std
            ]()
        else:
            self.bias = Tensor(shape(out_neurons)).initialize["zeros", 0.0]()

    fn forward_leaky_relu(self, x: Tensor) raises -> Tensor:
        return (x @ self.W + (self.bias * Float32(self.use_bias))).compute_activation[
            "lrelu", self.alpha
        ]()

    # TODO: Test
    # Dropout
    fn init_dropout(
        inout self,
    ) raises:
        self.W = self.bias = Tensor(shape(0))

    fn forward_dropout(self, x: Tensor) raises -> Tensor:
        return x.dropout[dropout_rate, noise_shape]()
