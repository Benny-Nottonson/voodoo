from tensor import TensorShape
from initializer import Initializer, XavierNormal
from regularizer import Regularizer, Empty
from activation import ActivationLayer, ReLU
from utilities import transposition, axis_sum

trait Layer:
    ...


struct HiddenLayers:
    ...


"""
struct Dense(Layer):
    var units: Int
    var activation: ActivationLayer
    var use_bias: Bool
    var kernel_initializer: Initializer
    var bias_initializer: Initializer
    var kernel_regularizer: Regularizer
    var bias_regularizer: Regularizer
    var activity_regularizer: Regularizer
    var kernel_constraint: Regularizer
    var bias_constraint: Regularizer

    fn __init__(inout self,
        units: Int,
        activation: ActivationLayer = ReLU(),
        use_bias: Bool = True,
        kernel_initializer: Initializer = XavierNormal(),
        bias_initializer: Initializer = XavierNormal(),
        kernel_regularizer: Regularizer = Empty(),
        bias_regularizer: Regularizer = Empty(),
        activity_regularizer: Regularizer = Empty(),
        kernel_constraint: Regularizer = Empty(),
        bias_constraint: Regularizer = Empty(),
    ):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
"""


@value
struct StaticDense[T: DType](Layer):
    var input_size: Int
    var output_size: Int
    var activation: ReLU
    var use_bias: Bool
    var kernel: Tensor[T]
    var bias: Tensor[T]

    fn __init__(
        inout self,
        input_size: Int,
        output_size: Int,
        activation: ReLU = ReLU(),
        use_bias: Bool = True,
        kernel_initializer: XavierNormal = XavierNormal(),
        bias_initializer: XavierNormal = XavierNormal(),
        # TODO
        kernel_regularizer: Empty = Empty(),
        bias_regularizer: Empty = Empty(),
        kernel_constraint: Empty = Empty(),
        bias_constraint: Empty = Empty(),
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.use_bias = use_bias
        self.kernel = kernel_initializer.initialize(
            Tensor[T](TensorShape(input_size, output_size))
        )
        self.bias = bias_initializer.initialize(Tensor[T](TensorShape(output_size)), 0)

    fn forward(self, input: Tensor[T]) raises -> Tensor[T]:
        return self.activation.forward(input * self.kernel + self.bias * self.use_bias)

    fn backward(inout self, input: Tensor[T], grad: Tensor[T]) raises -> Tensor[T]:
        let delta = grad * self.activation.deriv(input * self.kernel + self.bias)
        self.kernel = self.kernel - transposition(input) * delta

        if self.use_bias:
            self.bias = self.bias - axis_sum(delta, 0)

        return delta * transposition(self.kernel)
        
