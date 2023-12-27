from tensor import TensorShape
from initializer import Initializer
from regularizer import Regularizer
from activation import Activation
from loss import Loss
from utilities import transposition, axis_sum


@value
struct ActivationLayer[
    T: DType,
    activation_type: String,
    input_shape: TensorShape,
]:
    var forward: fn(Tensor[T]) -> Tensor[T]
    var backward: fn(Tensor[T]) -> Tensor[T]

    fn __init__(inout self) raises -> None:
        self.forward = Activation[T, activation_type]().forward
        self.backward = Activation[T, activation_type]().deriv

@value
struct Dense[
    T: DType,
    input_shape: TensorShape,
    output_shape: TensorShape,
    initializer_type: String,
    regularizer_type: String,
    activation_type: String,
]:
    var weights: Tensor[T]
    var biases: Tensor[T]
    var forward: fn(Tensor[T]) -> Tensor[T]
    var backward: fn(Tensor[T]) -> Tensor[T]

    fn __init__(inout self) raises -> None:
        let weightShape = TensorShape(input_shape[0], output_shape[0])
        self.weights = Initializer[T, initializer_type]().initialize(weightShape)
        self.biases = Initializer[T, initializer_type]().initialize(output_shape)
        self.forward = ActivationLayer[T, activation_type, input_shape]().forward
        self.backward = ActivationLayer[T, activation_type, input_shape]().backward

    fn update(inout self, weights: Tensor[T], biases: Tensor[T]) -> None:
        self.weights = weights
        self.biases = biases