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
]:  
    var name: String
    var activation: Activation
    var weights: Tensor[T]
    var bias: Tensor[T]

    var input: Tensor[T]
    var z: Tensor[T]
    var dw: Tensor[T]
    var db: Tensor[T]

    fn __init__(
        inout self,
        name: String = "dense",
        activation: Activation = Activation("relu"),
        kernelInitializer: Initializer = Initializer("xavier"),
        biasInitializer: Initializer = Initializer(""),
        kernelRegularizer: Regularizer = Regularizer(""),
        biasRegularizer: Regularizer = Regularizer(""),
    ):
        self.name = name
        self.activation = activation
        self.weights = kernelInitializer.initialize(Tensor[T](TensorShape(in_features, out_features)))
        self.bias = biasInitializer.initialize(Tensor[T](TensorShape(in_features, out_features)))
        self.input = Tensor[T](TensorShape(self.weights.shape()[0]))
        self.z = Tensor[T](TensorShape(out_features, 1))
        self.dw = Initializer("").initialize[T](Tensor[T](TensorShape(in_features, out_features)), 1)
        self.db = Initializer("").initialize[T](Tensor[T](TensorShape(in_features, out_features)), 1)

    fn summary(self):
        let n_params = self.weights.shape()[0] * self.weights.shape()[1] + self.bias.shape()[1] 
        let output_shape = self.bias.shape()[1]
        print("Dense Layer: " + self.name)
        print("Activation: " + self.activation.name)
        print("Output Shape: " + String(output_shape))
        print("Number of Parameters: " + String(n_params))

    fn forward(inout self, input: Tensor[T]) raises -> Tensor[T]:
        self.input = input
        let z = input * self.weights + self.bias
        self.z = z
        return self.activation.forward(z)

    fn backward(inout self, l: Loss[T]) raises -> Tensor[T]:
        let scalar = SIMD[T, 1](self.weights.shape()[0])
        let d = l.delta
        let dz = self.activation.deriv(self.z) * d
        self.dw = transposition(self.input) * dz / scalar
        for i in range(self.db.shape()[0]):
            self.db[i] = axis_sum(dz, 0)[i] / scalar
        self.update_weights(l.value)
        return dz * transposition(self.weights)

    fn update_weights(inout self, lr: SIMD[T, 1]) raises:
        self.weights = self.weights - lr * self.dw
        self.bias = self.bias - lr * self.db
