from tensor import TensorShape
from initializer import Initializer
from regularizer import Regularizer
from activation import Activation
from loss import Loss
from utilities import transposition, axis_sum

@value
struct DenseLayer[
    T: DType,
    in_features: Int,
    out_features: Int,
    activation_type: String = "relu",
    kernel_initializer_type: String = "xavier_normal",
    bias_initializer_type: String = "zeros",
    kernel_regularizer_type: String = "l2",
    bias_regularizer_type: String = "l2",
]:  
    var name: String
    var activation: Activation[T, activation_type]
    var weights: Tensor[T]
    var bias: Tensor[T]

    var input: Tensor[T]
    var z: Tensor[T]
    var dw: Tensor[T]
    var db: Tensor[T]

    fn __init__(
        inout self,
        name: String = "dense",
    ) raises:
        self.name = name
        self.activation = Activation[T, activation_type]()
        self.weights = Tensor[T](TensorShape(in_features, out_features))
        Initializer[T, kernel_initializer_type]().initialize(self.weights)
        self.bias = Tensor[T](TensorShape(1, out_features))
        Initializer[T, bias_initializer_type]().initialize(self.bias)
        self.input = Tensor[T](TensorShape(1, in_features))
        self.z = Tensor[T](TensorShape(out_features, 1))
        self.dw = Tensor[T](TensorShape(in_features, out_features))
        self.db = Tensor[T](TensorShape(1, out_features))

    fn summary(self):
        print("Dense Layer: ", self.name)
        print("Activation: ", self.activation.name)
        print("Weights: ", self.weights.shape())
        print("Bias: ", self.bias.shape())

    fn forward(inout self, input: Tensor[T]) raises -> Tensor[T]:
        self.input = input
        self.z = input * self.weights + self.bias
        return self.activation.forward(self.z)

    fn backward(inout self, l: Loss[T]) raises -> Tensor[T]:
        let dout = self.activation.deriv(l.delta)
        self.dw = transposition(self.input) * dout
        self.db = axis_sum(dout, 0)
        return dout * transposition(self.weights)

    fn update(inout self, lr: SIMD[T, 1]) raises:
        self.weights = self.weights - lr * self.dw
        self.bias = self.bias - lr * self.db
