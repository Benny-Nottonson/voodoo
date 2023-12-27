from tensor import TensorShape
from builtin.debug_assert import debug_assert
from math import abs

alias T = DType.float64


fn near_equal(a: Float64, b: Float64) -> Bool:
    return abs(a - b) < 1e-6


fn test_activation() raises:
    from activation import Activation

    let x1 = Tensor[T](TensorShape(1), 1.0)
    let x2 = Tensor[T](TensorShape(1), -10.0)
    let x3 = Tensor[T](TensorShape(1), 0.0)
    let x4 = Tensor[T](TensorShape(1), 15.0)
    let x5 = Tensor[T](TensorShape(1), -2.0)

    let sigmoid = Activation[T, "sigmoid"]()

    debug_assert(
        near_equal(sigmoid.forward(x1)[0], 0.7310585786300049), "sigmoid(1.0) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x2)[0], 4.5397868702434395e-05),
        "sigmoid(-10.0) failed",
    )
    debug_assert(near_equal(sigmoid.forward(x3)[0], 0.5), "sigmoid(0.0) failed")
    debug_assert(
        near_equal(sigmoid.forward(x4)[0], 0.999999694097773), "sigmoid(15.0) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x5)[0], 0.11920292202211755), "sigmoid(-2.0) failed"
    )

    let relu = Activation[T, "relu"]()

    debug_assert(near_equal(relu.forward(x1)[0], 1.0), "relu(1.0) failed")
    debug_assert(near_equal(relu.forward(x2)[0], 0.0), "relu(-10.0) failed")
    debug_assert(near_equal(relu.forward(x3)[0], 0.0), "relu(0.0) failed")
    debug_assert(near_equal(relu.forward(x4)[0], 15.0), "relu(15.0) failed")
    debug_assert(near_equal(relu.forward(x5)[0], 0.0), "relu(-2.0) failed")

    let tanh = Activation[T, "tanh"]()

    debug_assert(
        near_equal(tanh.forward(x1)[0], 0.7615941559557649), "tanh(1.0) failed"
    )
    debug_assert(
        near_equal(tanh.forward(x2)[0], -0.9999999958776927), "tanh(-10.0) failed"
    )
    debug_assert(near_equal(tanh.forward(x3)[0], 0.0), "tanh(0.0) failed")
    debug_assert(
        near_equal(tanh.forward(x4)[0], 0.9999999999998128), "tanh(15.0) failed"
    )
    debug_assert(
        near_equal(tanh.forward(x5)[0], -0.9640275800758169), "tanh(-2.0) failed"
    )

    let linear = Activation[T, "linear"]()

    debug_assert(near_equal(linear.forward(x1)[0], 1.0), "linear(1.0) failed")
    debug_assert(near_equal(linear.forward(x2)[0], -10.0), "linear(-10.0) failed")
    debug_assert(near_equal(linear.forward(x3)[0], 0.0), "linear(0.0) failed")
    debug_assert(near_equal(linear.forward(x4)[0], 15.0), "linear(15.0) failed")
    debug_assert(near_equal(linear.forward(x5)[0], -2.0), "linear(-2.0) failed")


fn test_initializer() raises:
    from initializer import Initializer

    let x1 = Tensor[T](TensorShape(5))
    let x2 = Tensor[T](TensorShape(5))
    let x3 = Tensor[T](TensorShape(5))
    let x4 = Tensor[T](TensorShape(5))
    let x5 = Tensor[T](TensorShape(5))

    let randomUniform = Initializer[T, "random_uniform"]()

    randomUniform.initialize(x1)
    randomUniform.initialize(x2)
    randomUniform.initialize(x3)
    randomUniform.initialize(x4)
    randomUniform.initialize(x5)

    debug_assert(x1[0] != x2[0], "random_uniform failed")
    debug_assert(x1[1] != x2[1], "random_uniform failed")
    debug_assert(x1[2] != x2[2], "random_uniform failed")
    debug_assert(x1[3] != x2[3], "random_uniform failed")
    debug_assert(x1[4] != x2[4], "random_uniform failed")

    debug_assert(x1[0] != x3[0], "random_uniform failed")
    debug_assert(x1[1] != x3[1], "random_uniform failed")
    debug_assert(x1[2] != x3[2], "random_uniform failed")
    debug_assert(x1[3] != x3[3], "random_uniform failed")
    debug_assert(x1[4] != x3[4], "random_uniform failed")

    debug_assert(x1[0] != x4[0], "random_uniform failed")
    debug_assert(x1[1] != x4[1], "random_uniform failed")
    debug_assert(x1[2] != x4[2], "random_uniform failed")
    debug_assert(x1[3] != x4[3], "random_uniform failed")
    debug_assert(x1[4] != x4[4], "random_uniform failed")

    debug_assert(x1[0] != x5[0], "random_uniform failed")
    debug_assert(x1[1] != x5[1], "random_uniform failed")
    debug_assert(x1[2] != x5[2], "random_uniform failed")
    debug_assert(x1[3] != x5[3], "random_uniform failed")
    debug_assert(x1[4] != x5[4], "random_uniform failed")

    let xavierNormal = Initializer[T, "xavier_normal"]()

    xavierNormal.initialize(x1)
    xavierNormal.initialize(x2)
    xavierNormal.initialize(x3)
    xavierNormal.initialize(x4)
    xavierNormal.initialize(x5)

    debug_assert(x1[0] != x2[0], "xavier_normal failed")
    debug_assert(x1[1] != x2[1], "xavier_normal failed")
    debug_assert(x1[2] != x2[2], "xavier_normal failed")
    debug_assert(x1[3] != x2[3], "xavier_normal failed")
    debug_assert(x1[4] != x2[4], "xavier_normal failed")

    debug_assert(x1[0] != x3[0], "xavier_normal failed")
    debug_assert(x1[1] != x3[1], "xavier_normal failed")
    debug_assert(x1[2] != x3[2], "xavier_normal failed")
    debug_assert(x1[3] != x3[3], "xavier_normal failed")
    debug_assert(x1[4] != x3[4], "xavier_normal failed")

    debug_assert(x1[0] != x4[0], "xavier_normal failed")
    debug_assert(x1[1] != x4[1], "xavier_normal failed")
    debug_assert(x1[2] != x4[2], "xavier_normal failed")
    debug_assert(x1[3] != x4[3], "xavier_normal failed")
    debug_assert(x1[4] != x4[4], "xavier_normal failed")

    debug_assert(x1[0] != x5[0], "xavier_normal failed")
    debug_assert(x1[1] != x5[1], "xavier_normal failed")
    debug_assert(x1[2] != x5[2], "xavier_normal failed")
    debug_assert(x1[3] != x5[3], "xavier_normal failed")
    debug_assert(x1[4] != x5[4], "xavier_normal failed")

    let zeros = Initializer[T, "zeros"]()

    zeros.initialize(x1)
    zeros.initialize(x2)

    debug_assert(x1[0] == 0.0, "zeros failed")
    debug_assert(x1[1] == 0.0, "zeros failed")
    debug_assert(x1[2] == 0.0, "zeros failed")
    debug_assert(x1[3] == 0.0, "zeros failed")
    debug_assert(x1[4] == 0.0, "zeros failed")

    debug_assert(x2[0] == 0.0, "zeros failed")
    debug_assert(x2[1] == 0.0, "zeros failed")
    debug_assert(x2[2] == 0.0, "zeros failed")
    debug_assert(x2[3] == 0.0, "zeros failed")
    debug_assert(x2[4] == 0.0, "zeros failed")


fn test_layer() raises:
    from layer import DenseLayer

    var dense = DenseLayer[T, 4, 1]()

    let x1 = Tensor[T](TensorShape(4), 1.0)
    let x2 = Tensor[T](TensorShape(4), 2.0)
    let x3 = Tensor[T](TensorShape(4), 3.0)
    let x4 = Tensor[T](TensorShape(4), 4.0)

    let y1 = Tensor[T](TensorShape(1), 1.0)
    let y2 = Tensor[T](TensorShape(1), 2.0)
    let y3 = Tensor[T](TensorShape(1), 3.0)

    debug_assert(near_equal(dense.forward(x1)[0], 0.0), "dense(1.0) failed")
    debug_assert(near_equal(dense.forward(x2)[0], 0.0), "dense(2.0) failed")
    debug_assert(near_equal(dense.forward(x3)[0], 0.0), "dense(3.0) failed")
    debug_assert(near_equal(dense.forward(x4)[0], 0.0), "dense(4.0) failed")


fn test_loss() raises:
    from loss import LossFunction

    let x1 = Tensor[T](TensorShape(5), 1.0)
    let x2 = Tensor[T](TensorShape(5), 2.0)
    let x3 = Tensor[T](TensorShape(5), 3.0)
    let x4 = Tensor[T](TensorShape(5), 4.0)
    let x5 = Tensor[T](TensorShape(5), 5.0)

    let y1 = Tensor[T](TensorShape(5), 1.0)
    let y2 = Tensor[T](TensorShape(5), 3.0)
    let y3 = Tensor[T](TensorShape(5), 5.0)
    let y4 = Tensor[T](TensorShape(5), 7.0)
    let y5 = Tensor[T](TensorShape(5), 9.0)

    var mse = LossFunction[T, "mse"]()

    debug_assert(near_equal(mse.calculate(x1, y1).value, 0.0), "mse(1.0, 1.0) failed")
    debug_assert(near_equal(mse.calculate(x2, y2).value, 2.0), "mse(2.0, 3.0) failed")
    debug_assert(near_equal(mse.calculate(x3, y3).value, 8.0), "mse(3.0, 5.0) failed")
    debug_assert(near_equal(mse.calculate(x4, y4).value, 18.0), "mse(4.0, 7.0) failed")
    debug_assert(near_equal(mse.calculate(x5, y5).value, 32.0), "mse(5.0, 9.0) failed")


fn main() raises:
    test_activation()
    print("Activation all tests passed")
    test_initializer()
    print("Initializer all tests passed")
    test_layer()
    print("Layer all tests passed")
    test_loss()
    print("Loss all tests passed")
