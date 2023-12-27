from tensor import TensorShape
from builtin.debug_assert import debug_assert
from math import abs
from random import rand
from memory import memset

alias T = DType.float64


fn test_activation() raises:
    from activation import Activation

    @noncapturing
    fn near_equal(a: Float64, b: Float64) -> Bool:
        return abs(a - b) < 1e-6

    let x1 = Tensor[T](TensorShape(1), 1.0)
    let x2 = Tensor[T](TensorShape(1), 10)
    let x3 = Tensor[T](TensorShape(1), -1.0)
    let x4 = Tensor[T](TensorShape(1), -10)
    let x5 = Tensor[T](TensorShape(1), 0.0)
    let x6 = Tensor[T](TensorShape(1), 0.5)
    let x7 = Tensor[T](TensorShape(1), -0.5)
    let x8 = Tensor[T](TensorShape(1), 100)
    let x9 = Tensor[T](TensorShape(1), -100)

    let linear = Activation[T, "linear"]()
    let relu = Activation[T, "relu"]()
    let tanh = Activation[T, "tanh"]()
    let sigmoid = Activation[T, "sigmoid"]()
    let elu = Activation[T, "elu"]()
    let exponential = Activation[T, "exponential"]()
    let gelu = Activation[T, "gelu"]()
    let hard_sigmoid = Activation[T, "hard_sigmoid"]()
    let mish = Activation[T, "mish"]()
    let selu = Activation[T, "selu"]()
    let softplus = Activation[T, "softplus"]()
    let softsign = Activation[T, "softsign"]()

    debug_assert(near_equal(linear.forward(x1)[0], 1.0), "linear(1.0) failed")
    debug_assert(near_equal(linear.forward(x2)[0], 10.0), "linear(10.0) failed")
    debug_assert(near_equal(linear.forward(x3)[0], -1.0), "linear(-1.0) failed")
    debug_assert(near_equal(linear.forward(x4)[0], -10.0), "linear(-10.0) failed")
    debug_assert(near_equal(linear.forward(x5)[0], 0.0), "linear(0.0) failed")
    debug_assert(near_equal(linear.forward(x6)[0], 0.5), "linear(0.5) failed")
    debug_assert(near_equal(linear.forward(x7)[0], -0.5), "linear(-0.5) failed")
    debug_assert(near_equal(linear.forward(x8)[0], 100.0), "linear(100.0) failed")
    debug_assert(near_equal(linear.forward(x9)[0], -100.0), "linear(-100.0) failed")

    debug_assert(near_equal(relu.forward(x1)[0], 1.0), "relu(1.0) failed")
    debug_assert(near_equal(relu.forward(x2)[0], 10.0), "relu(10.0) failed")
    debug_assert(near_equal(relu.forward(x3)[0], 0.0), "relu(-1.0) failed")
    debug_assert(near_equal(relu.forward(x4)[0], 0.0), "relu(-10.0) failed")
    debug_assert(near_equal(relu.forward(x5)[0], 0.0), "relu(0.0) failed")
    debug_assert(near_equal(relu.forward(x6)[0], 0.5), "relu(0.5) failed")
    debug_assert(near_equal(relu.forward(x7)[0], 0.0), "relu(-0.5) failed")
    debug_assert(near_equal(relu.forward(x8)[0], 100.0), "relu(100.0) failed")
    debug_assert(near_equal(relu.forward(x9)[0], 0.0), "relu(-100.0) failed")

    debug_assert(
        near_equal(tanh.forward(x1)[0], 0.7615941559557649), "tanh(1.0) failed"
    )
    debug_assert(near_equal(tanh.forward(x2)[0], 1.0), "tanh(10.0) failed")
    debug_assert(
        near_equal(tanh.forward(x3)[0], -0.7615941559557649), "tanh(-1.0) failed"
    )
    debug_assert(near_equal(tanh.forward(x4)[0], -1.0), "tanh(-10.0) failed")
    debug_assert(near_equal(tanh.forward(x5)[0], 0.0), "tanh(0.0) failed")
    debug_assert(
        near_equal(tanh.forward(x6)[0], 0.46211715726000974), "tanh(0.5) failed"
    )
    debug_assert(
        near_equal(tanh.forward(x7)[0], -0.46211715726000974), "tanh(-0.5) failed"
    )
    debug_assert(near_equal(tanh.forward(x8)[0], 1.0), "tanh(100.0) failed")
    debug_assert(near_equal(tanh.forward(x9)[0], -1.0), "tanh(-100.0) failed")

    debug_assert(
        near_equal(sigmoid.forward(x1)[0], 0.7310585786300049), "sigmoid(1.0) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x2)[0], 0.9999546021312976), "sigmoid(10.0) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x3)[0], 0.2689414213699951), "sigmoid(-1.0) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x4)[0], 4.5397868702434395e-05),
        "sigmoid(-10.0) failed",
    )
    debug_assert(near_equal(sigmoid.forward(x5)[0], 0.5), "sigmoid(0.0) failed")
    debug_assert(
        near_equal(sigmoid.forward(x6)[0], 0.6224593312018546), "sigmoid(0.5) failed"
    )
    debug_assert(
        near_equal(sigmoid.forward(x7)[0], 0.3775406687981454), "sigmoid(-0.5) failed"
    )
    debug_assert(near_equal(sigmoid.forward(x8)[0], 1.0), "sigmoid(100.0) failed")
    debug_assert(
        near_equal(sigmoid.forward(x9)[0], 3.7200759760208356e-44),
        "sigmoid(-100.0) failed",
    )

    debug_assert(near_equal(elu.forward(x1)[0], 1.0), "elu(1.0) failed")
    debug_assert(near_equal(elu.forward(x2)[0], 10.0), "elu(10.0) failed")
    debug_assert(
        near_equal(elu.forward(x3)[0], -0.6321205588285577), "elu(-1.0) failed"
    )
    debug_assert(
        near_equal(elu.forward(x4)[0], -0.9999546000702375), "elu(-10.0) failed"
    )
    debug_assert(near_equal(elu.forward(x5)[0], 0.0), "elu(0.0) failed")
    debug_assert(near_equal(elu.forward(x6)[0], 0.5), "elu(0.5) failed")
    debug_assert(
        near_equal(elu.forward(x7)[0], -0.3934693402873666), "elu(-0.5) failed"
    )
    debug_assert(near_equal(elu.forward(x8)[0], 100.0), "elu(100.0) failed")
    debug_assert(near_equal(elu.forward(x9)[0], -1.0), "elu(-100.0) failed")

    debug_assert(
        near_equal(exponential.forward(x1)[0], 2.718281828459045),
        "exponential(1.0) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x2)[0], 22026.465794806718),
        "exponential(10.0) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x3)[0], 0.36787944117144233),
        "exponential(-1.0) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x4)[0], 4.5399929762484854e-05),
        "exponential(-10.0) failed",
    )
    debug_assert(near_equal(exponential.forward(x5)[0], 1.0), "exponential(0.0) failed")
    debug_assert(
        near_equal(exponential.forward(x6)[0], 1.6487212707001282),
        "exponential(0.5) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x7)[0], 0.6065306597126334),
        "exponential(-0.5) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x8)[0], 2.6881171418161356e43),
        "exponential(100.0) failed",
    )
    debug_assert(
        near_equal(exponential.forward(x9)[0], 3.7200759760208356e-44),
        "exponential(-100.0) failed",
    )

    debug_assert(near_equal(gelu.forward(x1)[0], 0.841192), "gelu(1.0) failed")
    debug_assert(near_equal(gelu.forward(x2)[0], 10.0), "gelu(10.0) failed")
    debug_assert(near_equal(gelu.forward(x3)[0], -0.158808), "gelu(-1.0) failed")
    debug_assert(near_equal(gelu.forward(x4)[0], -10.0), "gelu(-10.0) failed")
    debug_assert(near_equal(gelu.forward(x5)[0], 0.0), "gelu(0.0) failed")
    debug_assert(near_equal(gelu.forward(x6)[0], 0.345703), "gelu(0.5) failed")
    debug_assert(near_equal(gelu.forward(x7)[0], -0.345703), "gelu(-0.5) failed")
    debug_assert(near_equal(gelu.forward(x8)[0], 100.0), "gelu(100.0) failed")
    debug_assert(near_equal(gelu.forward(x9)[0], -100.0), "gelu(-100.0) failed")

    debug_assert(
        near_equal(hard_sigmoid.forward(x1)[0], 0.7), "hard_sigmoid(1.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x2)[0], 1.0), "hard_sigmoid(10.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x3)[0], 0.3), "hard_sigmoid(-1.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x4)[0], 0.0), "hard_sigmoid(-10.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x5)[0], 0.5), "hard_sigmoid(0.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x6)[0], 0.65), "hard_sigmoid(0.5) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x7)[0], 0.35), "hard_sigmoid(-0.5) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x8)[0], 1.0), "hard_sigmoid(100.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.forward(x9)[0], 0.0), "hard_sigmoid(-100.0) failed"
    )

    debug_assert(near_equal(mish.forward(x1)[0], 0.865098), "mish(1.0) failed")
    debug_assert(near_equal(mish.forward(x2)[0], 10.0), "mish(10.0) failed")
    debug_assert(near_equal(mish.forward(x3)[0], -0.303401), "mish(-1.0) failed")
    debug_assert(near_equal(mish.forward(x4)[0], -10.0), "mish(-10.0) failed")
    debug_assert(near_equal(mish.forward(x5)[0], 0.0), "mish(0.0) failed")
    debug_assert(near_equal(mish.forward(x6)[0], 0.462117), "mish(0.5) failed")
    debug_assert(near_equal(mish.forward(x7)[0], -0.462117), "mish(-0.5) failed")
    debug_assert(near_equal(mish.forward(x8)[0], 100.0), "mish(100.0) failed")
    debug_assert(near_equal(mish.forward(x9)[0], -100.0), "mish(-100.0) failed")

    debug_assert(near_equal(selu.forward(x1)[0], 1.050701), "selu(1.0) failed")
    debug_assert(near_equal(selu.forward(x2)[0], 10.50701), "selu(10.0) failed")
    debug_assert(near_equal(selu.forward(x3)[0], -1.111330), "selu(-1.0) failed")
    debug_assert(near_equal(selu.forward(x4)[0], -11.1133), "selu(-10.0) failed")
    debug_assert(near_equal(selu.forward(x5)[0], 0.0), "selu(0.0) failed")
    debug_assert(near_equal(selu.forward(x6)[0], 0.525350), "selu(0.5) failed")
    debug_assert(near_equal(selu.forward(x7)[0], -0.367875), "selu(-0.5) failed")
    debug_assert(near_equal(selu.forward(x8)[0], 105.0701), "selu(100.0) failed")
    debug_assert(near_equal(selu.forward(x9)[0], -111.133), "selu(-100.0) failed")

    debug_assert(
        near_equal(softplus.forward(x1)[0], 1.3132616875182228), "softplus(1.0) failed"
    )
    debug_assert(
        near_equal(softplus.forward(x2)[0], 10.000045398899218), "softplus(10.0) failed"
    )
    debug_assert(
        near_equal(softplus.forward(x3)[0], 0.31326168751822286),
        "softplus(-1.0) failed",
    )
    debug_assert(
        near_equal(softplus.forward(x4)[0], 4.5398899216870535e-05),
        "softplus(-10.0) failed",
    )
    debug_assert(
        near_equal(softplus.forward(x5)[0], 0.6931471805599453), "softplus(0.0) failed"
    )
    debug_assert(
        near_equal(softplus.forward(x6)[0], 0.9740769841804061), "softplus(0.5) failed"
    )
    debug_assert(
        near_equal(softplus.forward(x7)[0], 0.47407698418040613),
        "softplus(-0.5) failed",
    )
    debug_assert(near_equal(softplus.forward(x8)[0], 100.0), "softplus(100.0) failed")
    debug_assert(near_equal(softplus.forward(x9)[0], 0.0), "softplus(-100.0) failed")

    debug_assert(near_equal(softsign.forward(x1)[0], 0.5), "softsign(1.0) failed")
    debug_assert(
        near_equal(softsign.forward(x2)[0], 0.9090909090909091), "softsign(10.0) failed"
    )
    debug_assert(near_equal(softsign.forward(x3)[0], -0.5), "softsign(-1.0) failed")
    debug_assert(
        near_equal(softsign.forward(x4)[0], -0.9090909090909091),
        "softsign(-10.0) failed",
    )
    debug_assert(near_equal(softsign.forward(x5)[0], 0.0), "softsign(0.0) failed")
    debug_assert(
        near_equal(softsign.forward(x6)[0], 0.3333333333333333), "softsign(0.5) failed"
    )
    debug_assert(
        near_equal(softsign.forward(x7)[0], -0.3333333333333333),
        "softsign(-0.5) failed",
    )
    debug_assert(
        near_equal(softsign.forward(x8)[0], 0.9900990099009901),
        "softsign(100.0) failed",
    )
    debug_assert(
        near_equal(softsign.forward(x9)[0], -0.9900990099009901),
        "softsign(-100.0) failed",
    )

    debug_assert(near_equal(linear.deriv(x1)[0], 1.0), "linear.deriv(1.0) failed")
    debug_assert(near_equal(linear.deriv(x2)[0], 1.0), "linear.deriv(10.0) failed")
    debug_assert(near_equal(linear.deriv(x3)[0], 1.0), "linear.deriv(-1.0) failed")
    debug_assert(near_equal(linear.deriv(x4)[0], 1.0), "linear.deriv(-10.0) failed")
    debug_assert(near_equal(linear.deriv(x5)[0], 1.0), "linear.deriv(0.0) failed")
    debug_assert(near_equal(linear.deriv(x6)[0], 1.0), "linear.deriv(0.5) failed")
    debug_assert(near_equal(linear.deriv(x7)[0], 1.0), "linear.deriv(-0.5) failed")
    debug_assert(near_equal(linear.deriv(x8)[0], 1.0), "linear.deriv(100.0) failed")
    debug_assert(near_equal(linear.deriv(x9)[0], 1.0), "linear.deriv(-100.0) failed")

    debug_assert(near_equal(relu.deriv(x1)[0], 1.0), "relu.deriv(1.0) failed")
    debug_assert(near_equal(relu.deriv(x2)[0], 1.0), "relu.deriv(10.0) failed")
    debug_assert(near_equal(relu.deriv(x3)[0], 0.0), "relu.deriv(-1.0) failed")
    debug_assert(near_equal(relu.deriv(x4)[0], 0.0), "relu.deriv(-10.0) failed")
    debug_assert(near_equal(relu.deriv(x5)[0], 0.0), "relu.deriv(0.0) failed")
    debug_assert(near_equal(relu.deriv(x6)[0], 1.0), "relu.deriv(0.5) failed")
    debug_assert(near_equal(relu.deriv(x7)[0], 0.0), "relu.deriv(-0.5) failed")
    debug_assert(near_equal(relu.deriv(x8)[0], 1.0), "relu.deriv(100.0) failed")
    debug_assert(near_equal(relu.deriv(x9)[0], 0.0), "relu.deriv(-100.0) failed")

    debug_assert(
        near_equal(tanh.deriv(x1)[0], 0.41997434161402614), "tanh.deriv(1.0) failed"
    )
    debug_assert(near_equal(tanh.deriv(x2)[0], 0.0), "tanh.deriv(10.0) failed")
    debug_assert(
        near_equal(tanh.deriv(x3)[0], 0.41997434161402614), "tanh.deriv(-1.0) failed"
    )
    debug_assert(near_equal(tanh.deriv(x4)[0], 0.0), "tanh.deriv(-10.0) failed")
    debug_assert(near_equal(tanh.deriv(x5)[0], 1.0), "tanh.deriv(0.0) failed")
    debug_assert(
        near_equal(tanh.deriv(x6)[0], 0.7864477329659274), "tanh.deriv(0.5) failed"
    )
    debug_assert(
        near_equal(tanh.deriv(x7)[0], 0.7864477329659274), "tanh.deriv(-0.5) failed"
    )
    debug_assert(near_equal(tanh.deriv(x8)[0], 0.0), "tanh.deriv(100.0) failed")
    debug_assert(near_equal(tanh.deriv(x9)[0], 0.0), "tanh.deriv(-100.0) failed")

    debug_assert(
        near_equal(sigmoid.deriv(x1)[0], 0.19661193324148185),
        "sigmoid.deriv(1.0) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x2)[0], 4.5395807735951673e-05),
        "sigmoid.deriv(10.0) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x3)[0], 0.19661193324148185),
        "sigmoid.deriv(-1.0) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x4)[0], 4.5395807735951673e-05),
        "sigmoid.deriv(-10.0) failed",
    )
    debug_assert(near_equal(sigmoid.deriv(x5)[0], 0.25), "sigmoid.deriv(0.0) failed")
    debug_assert(
        near_equal(sigmoid.deriv(x6)[0], 0.2350037122015945),
        "sigmoid.deriv(0.5) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x7)[0], 0.2350037122015945),
        "sigmoid.deriv(-0.5) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x8)[0], 3.7200759760208356e-44),
        "sigmoid.deriv(100.0) failed",
    )
    debug_assert(
        near_equal(sigmoid.deriv(x9)[0], 3.7200759760208356e-44),
        "sigmoid.deriv(-100.0) failed",
    )

    debug_assert(near_equal(elu.deriv(x1)[0], 1.0), "elu.deriv(1.0) failed")
    debug_assert(near_equal(elu.deriv(x2)[0], 1.0), "elu.deriv(10.0) failed")
    debug_assert(
        near_equal(elu.deriv(x3)[0], 0.36787944117144233), "elu.deriv(-1.0) failed"
    )
    debug_assert(
        near_equal(elu.deriv(x4)[0], 4.5399929762484854e-05), "elu.deriv(-10.0) failed"
    )
    debug_assert(near_equal(elu.deriv(x5)[0], 1.0), "elu.deriv(0.0) failed")
    debug_assert(
        near_equal(elu.deriv(x6)[0], 1.1331484530668263), "elu.deriv(0.5) failed"
    )
    debug_assert(
        near_equal(elu.deriv(x7)[0], 0.6224593312018546), "elu.deriv(-0.5) failed"
    )
    debug_assert(near_equal(elu.deriv(x8)[0], 1.0), "elu.deriv(100.0) failed")
    debug_assert(near_equal(elu.deriv(x9)[0], 0.0), "elu.deriv(-100.0) failed")

    debug_assert(
        near_equal(exponential.deriv(x1)[0], 2.718281828459045),
        "exponential.deriv(1.0) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x2)[0], 22026.465794806718),
        "exponential.deriv(10.0) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x3)[0], 0.36787944117144233),
        "exponential.deriv(-1.0) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x4)[0], 4.5399929762484854e-05),
        "exponential.deriv(-10.0) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x5)[0], 1.0), "exponential.deriv(0.0) failed"
    )
    debug_assert(
        near_equal(exponential.deriv(x6)[0], 1.6487212707001282),
        "exponential.deriv(0.5) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x7)[0], 0.6065306597126334),
        "exponential.deriv(-0.5) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x8)[0], 2.6881171418161356e43),
        "exponential.deriv(100.0) failed",
    )
    debug_assert(
        near_equal(exponential.deriv(x9)[0], 3.7200759760208356e-44),
        "exponential.deriv(-100.0) failed",
    )

    debug_assert(
        near_equal(gelu.deriv(x1)[0], 0.41997434161402614), "gelu.deriv(1.0) failed"
    )
    debug_assert(near_equal(gelu.deriv(x2)[0], 1.0), "gelu.deriv(10.0) failed")
    debug_assert(
        near_equal(gelu.deriv(x3)[0], 0.41997434161402614), "gelu.deriv(-1.0) failed"
    )
    debug_assert(near_equal(gelu.deriv(x4)[0], 1.0), "gelu.deriv(-10.0) failed")
    debug_assert(near_equal(gelu.deriv(x5)[0], 0.5), "gelu.deriv(0.0) failed")
    debug_assert(near_equal(gelu.deriv(x6)[0], 0.345703), "gelu.deriv(0.5) failed")
    debug_assert(near_equal(gelu.deriv(x7)[0], 0.345703), "gelu.deriv(-0.5) failed")
    debug_assert(near_equal(gelu.deriv(x8)[0], 1.0), "gelu.deriv(100.0) failed")
    debug_assert(near_equal(gelu.deriv(x9)[0], 1.0), "gelu.deriv(-100.0) failed")

    debug_assert(
        near_equal(hard_sigmoid.deriv(x1)[0], 0.2), "hard_sigmoid.deriv(1.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x2)[0], 0.0), "hard_sigmoid.deriv(10.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x3)[0], 0.2), "hard_sigmoid.deriv(-1.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x4)[0], 0.0), "hard_sigmoid.deriv(-10.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x5)[0], 0.25), "hard_sigmoid.deriv(0.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x6)[0], 0.3), "hard_sigmoid.deriv(0.5) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x7)[0], 0.3), "hard_sigmoid.deriv(-0.5) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x8)[0], 0.0), "hard_sigmoid.deriv(100.0) failed"
    )
    debug_assert(
        near_equal(hard_sigmoid.deriv(x9)[0], 0.0), "hard_sigmoid.deriv(-100.0) failed"
    )

    debug_assert(
        near_equal(mish.deriv(x1)[0], 0.41997434161402614), "mish.deriv(1.0) failed"
    )
    debug_assert(near_equal(mish.deriv(x2)[0], 1.0), "mish.deriv(10.0) failed")
    debug_assert(
        near_equal(mish.deriv(x3)[0], 0.41997434161402614), "mish.deriv(-1.0) failed"
    )
    debug_assert(near_equal(mish.deriv(x4)[0], 1.0), "mish.deriv(-10.0) failed")
    debug_assert(near_equal(mish.deriv(x5)[0], 0.5), "mish.deriv(0.0) failed")
    debug_assert(near_equal(mish.deriv(x6)[0], 0.462117), "mish.deriv(0.5) failed")
    debug_assert(near_equal(mish.deriv(x7)[0], 0.462117), "mish.deriv(-0.5) failed")
    debug_assert(near_equal(mish.deriv(x8)[0], 1.0), "mish.deriv(100.0) failed")
    debug_assert(near_equal(mish.deriv(x9)[0], 1.0), "mish.deriv(-100.0) failed")

    debug_assert(near_equal(selu.deriv(x1)[0], 1.050701), "selu.deriv(1.0) failed")
    debug_assert(near_equal(selu.deriv(x2)[0], 1.050701), "selu.deriv(10.0) failed")
    debug_assert(
        near_equal(selu.deriv(x3)[0], 0.36787944117144233), "selu.deriv(-1.0) failed"
    )
    debug_assert(
        near_equal(selu.deriv(x4)[0], 0.36787944117144233), "selu.deriv(-10.0) failed"
    )
    debug_assert(near_equal(selu.deriv(x5)[0], 1.0), "selu.deriv(0.0) failed")
    debug_assert(near_equal(selu.deriv(x6)[0], 1.050701), "selu.deriv(0.5) failed")
    debug_assert(
        near_equal(selu.deriv(x7)[0], 0.36787944117144233), "selu.deriv(-0.5) failed"
    )
    debug_assert(near_equal(selu.deriv(x8)[0], 1.050701), "selu.deriv(100.0) failed")
    debug_assert(
        near_equal(selu.deriv(x9)[0], 0.36787944117144233), "selu.deriv(-100.0) failed"
    )

    debug_assert(
        near_equal(softplus.deriv(x1)[0], 0.7310585786300049),
        "softplus.deriv(1.0) failed",
    )
    debug_assert(near_equal(softplus.deriv(x2)[0], 1.0), "softplus.deriv(10.0) failed")
    debug_assert(
        near_equal(softplus.deriv(x3)[0], 0.7310585786300049),
        "softplus.deriv(-1.0) failed",
    )
    debug_assert(near_equal(softplus.deriv(x4)[0], 1.0), "softplus.deriv(-10.0) failed")
    debug_assert(near_equal(softplus.deriv(x5)[0], 0.5), "softplus.deriv(0.0) failed")
    debug_assert(
        near_equal(softplus.deriv(x6)[0], 0.8807970779778823),
        "softplus.deriv(0.5) failed",
    )
    debug_assert(
        near_equal(softplus.deriv(x7)[0], 0.8807970779778823),
        "softplus.deriv(-0.5) failed",
    )
    debug_assert(near_equal(softplus.deriv(x8)[0], 1.0), "softplus.deriv(100.0) failed")
    debug_assert(
        near_equal(softplus.deriv(x9)[0], 0.0), "softplus.deriv(-100.0) failed"
    )

    debug_assert(near_equal(softsign.deriv(x1)[0], 0.25), "softsign.deriv(1.0) failed")
    debug_assert(
        near_equal(softsign.deriv(x2)[0], 0.00909090909090909),
        "softsign.deriv(10.0) failed",
    )
    debug_assert(near_equal(softsign.deriv(x3)[0], 0.25), "softsign.deriv(-1.0) failed")
    debug_assert(
        near_equal(softsign.deriv(x4)[0], 0.00909090909090909),
        "softsign.deriv(-10.0) failed",
    )
    debug_assert(near_equal(softsign.deriv(x5)[0], 0.25), "softsign.deriv(0.0) failed")
    debug_assert(
        near_equal(softsign.deriv(x6)[0], 0.1111111111111111),
        "softsign.deriv(0.5) failed",
    )
    debug_assert(
        near_equal(softsign.deriv(x7)[0], 0.1111111111111111),
        "softsign.deriv(-0.5) failed",
    )
    debug_assert(
        near_equal(softsign.deriv(x8)[0], 0.009900990099009901),
        "softsign.deriv(100.0) failed",
    )
    debug_assert(
        near_equal(softsign.deriv(x9)[0], 0.009900990099009901),
        "softsign.deriv(-100.0) failed",
    )


fn test_initializer() raises:
    from initializer import Initializer

    let shape = TensorShape(2, 2)

    let zeros = Initializer[T, "zeros"]()
    let ones = Initializer[T, "ones"]()
    let glorot_normal = Initializer[T, "glorot_normal"]()
    let glorot_uniform = Initializer[T, "glorot_uniform"]()
    let he_normal = Initializer[T, "he_normal"]()
    let he_uniform = Initializer[T, "he_uniform"]()
    let identity = Initializer[T, "identity"]()
    let lecun_normal = Initializer[T, "lecun_normal"]()
    let lecun_uniform = Initializer[T, "lecun_uniform"]()
    let random_normal = Initializer[T, "random_normal"]()
    let random_uniform = Initializer[T, "random_uniform"]()
    let truncated_normal = Initializer[T, "truncated_normal"]()

    let x0 = zeros.constant_initialize(shape, 5)
    let x1 = zeros.initialize(shape)
    let x2 = ones.initialize(shape)
    let x3 = glorot_normal.initialize(shape)
    let x4 = glorot_uniform.initialize(shape)
    let x5 = he_normal.initialize(shape)
    let x6 = he_uniform.initialize(shape)
    let x7 = identity.initialize(shape)
    let x8 = lecun_normal.initialize(shape)
    let x9 = lecun_uniform.initialize(shape)
    let x10 = random_normal.initialize(shape)
    let x11 = random_uniform.initialize(shape)
    let x12 = truncated_normal.initialize(shape)

    debug_assert(x0.shape() == shape, "zeros.constant_initialize shape failed")
    debug_assert(x0[0] == 5, "zeros.constant_initialize value failed")
    debug_assert(x0[1] == 5, "zeros.constant_initialize value failed")

    debug_assert(x1.shape() == shape, "zeros.initialize shape failed")
    debug_assert(x1[0] == 0, "zeros.initialize value failed")
    debug_assert(x1[1] == 0, "zeros.initialize value failed")

    debug_assert(x2.shape() == shape, "ones.initialize shape failed")
    debug_assert(x2[0] == 1, "ones.initialize value failed")
    debug_assert(x2[1] == 1, "ones.initialize value failed")

    debug_assert(x3.shape() == shape, "glorot_normal.initialize shape failed")
    debug_assert(x3[0] != x3[1], "glorot_normal.initialize value failed")

    debug_assert(x4.shape() == shape, "glorot_uniform.initialize shape failed")
    debug_assert(x4[0] != x4[1], "glorot_uniform.initialize value failed")

    debug_assert(x5.shape() == shape, "he_normal.initialize shape failed")
    debug_assert(x5[0] != x5[1], "he_normal.initialize value failed")

    debug_assert(x6.shape() == shape, "he_uniform.initialize shape failed")
    debug_assert(x6[0] != x6[1], "he_uniform.initialize value failed")

    debug_assert(x7.shape() == shape, "identity.initialize shape failed")
    debug_assert(x7[0] == 1, "identity.initialize value failed")

    debug_assert(x8.shape() == shape, "lecun_normal.initialize shape failed")
    debug_assert(x8[0] != x8[1], "lecun_normal.initialize value failed")

    debug_assert(x9.shape() == shape, "lecun_uniform.initialize shape failed")
    debug_assert(x9[0] != x9[1], "lecun_uniform.initialize value failed")

    debug_assert(x10.shape() == shape, "random_normal.initialize shape failed")
    debug_assert(x10[0] != x10[1], "random_normal.initialize value failed")

    debug_assert(x11.shape() == shape, "random_uniform.initialize shape failed")
    debug_assert(x11[0] != x11[1], "random_uniform.initialize value failed")

    debug_assert(x12.shape() == shape, "truncated_normal.initialize shape failed")
    debug_assert(x12[0] != x12[1], "truncated_normal.initialize value failed")


fn main() raises:
    test_activation()
    print("Activation all tests passed")
    test_initializer()
    print("Initializer all tests passed")
