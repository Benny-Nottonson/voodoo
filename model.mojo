from activation import *
from initializer import *
from layer import *
from loss import *
from optimizer import *
from regularizer import *
from inputs import *
from tensor import TensorShape
from random import random_float64


@value
struct SingleSequential[
    T: DType,
    in_features: Int,
    out_features: Int,
]:
    var loss_fn: MeanSquaredError[T]
    var layer: DenseLayer[T, in_features, out_features]

    fn forward(inout self, x: Tensor[T]) raises -> Tensor[T]:
        return self.layer.forward(x)

    fn backward(inout self, x: Tensor[T]) raises:
        let loss = self.loss_fn.calculate(x, self.forward(x))
        _ = self.layer.backward(loss)


fn main() raises:
    var m = SingleSequential[DType.float64, 2, 1,](
        loss_fn=MeanSquaredError[DType.float64](),
        layer=DenseLayer[DType.float64, 2, 1](
            activation=Activation("relu"),
        ),
    )

    var x = Tensor[DType.float64](
        TensorShape(2, 1),
    )
    var y = Tensor[DType.float64](
        TensorShape(1, 1),
    )

    for i in range(5):
        x[0][0] = random_float64(-100, 100)
        x[1][0] = random_float64(-100, 100)

        y[0][0] = (x[0][0] + x[1][0]) / 2

        m.backward(x)
        print(m.layer.weights)
