from math import exp, max, tanh, log, abs
from utilities import map

alias activationFunctions = VariadicList[StringLiteral](
    "linear",
    "relu",
    "tanh",
    "sigmoid",
    "elu",
    "exponential",
    "gelu",
    "hard_sigmoid",
    "mish",
    "selu",
    "softplus",
    "softsign",
)


@value
struct Activation[T: DType, name: String]:
    var forward: fn (Tensor[T]) -> Tensor[T]
    var deriv: fn (Tensor[T]) -> Tensor[T]

    fn __init__(inout self) raises:
        if name == "linear":
            self.forward = Linear.forward[T]
            self.deriv = Linear.deriv[T]
        elif name == "relu":
            self.forward = ReLU.forward[T]
            self.deriv = ReLU.deriv[T]
        elif name == "tanh":
            self.forward = Tanh.forward[T]
            self.deriv = Tanh.deriv[T]
        elif name == "sigmoid":
            self.forward = Sigmoid.forward[T]
            self.deriv = Sigmoid.deriv[T]
        elif name == "elu":
            self.forward = ELU.forward[T]
            self.deriv = ELU.deriv[T]
        elif name == "exponential":
            self.forward = Exponential.forward[T]
            self.deriv = Exponential.deriv[T]
        elif name == "gelu":
            self.forward = GELU.forward[T]
            self.deriv = GELU.deriv[T]
        elif name == "hard_sigmoid":
            self.forward = HardSigmoid.forward[T]
            self.deriv = HardSigmoid.deriv[T]
        elif name == "mish":
            self.forward = Mish.forward[T]
            self.deriv = Mish.deriv[T]
        elif name == "selu":
            self.forward = SELU.forward[T]
            self.deriv = SELU.deriv[T]
        elif name == "softplus":
            self.forward = Softplus.forward[T]
            self.deriv = Softplus.deriv[T]
        elif name == "softsign":
            self.forward = Softsign.forward[T]
            self.deriv = Softsign.deriv[T]
        else:
            raise Error("Invalid activation function name: " + name)


struct Linear:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Tensor[T](x.shape(), 1)


struct ReLU:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return max(x, SIMD[T, 1](0))

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return SIMD[T, 1](x > 0)

        return map[T](x, func)


struct Tanh:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return tanh(x)

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return 1 - tanh(x) ** 2

        return map[T](x, func)


struct Sigmoid:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return 1 / (1 + exp(-x))

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return exp(-x) / (1 + exp(-x)) ** 2

        return map[T](x, func)


struct ELU:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return x * (x > 0).cast[T]() + (exp(x) - 1) * (x <= 0).cast[T]()

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return (x > 0).cast[T]() + (exp(x) * (x <= 0).cast[T]())

        return map[T](x, func)


struct Exponential:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return exp(x)

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Exponential.forward[T](x)


struct GELU:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return x * (x > 0).cast[T]() + x * (x <= 0).cast[T]() * (
                0.5 * (1 + tanh(0.797885 * (x + 0.044715 * x**3)))
            )

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return (x > 0).cast[T]() + (
                0.797885 * (x + 0.044715 * x**3)
                + 0.797885 * x**3 * tanh(0.797885 * (x + 0.044715 * x**3))
                + 0.134145
                * x**2
                * (1 - tanh(0.797885 * (x + 0.044715 * x**3)) ** 2)
            ) * (x <= 0).cast[T]()

        return map[T](x, func)


struct HardSigmoid:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        return (x * 0.2 + 0.5).clip(0, 1)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return (x >= -2.5).cast[T]() * (x <= 2.5).cast[T]() * 0.2

        return map[T](x, func)


struct Mish:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return x * tanh(log(1 + exp(x)))

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            let s = tanh(log(1 + exp(x)))
            return x * (1 - s**2) + s

        return map[T](x, func)


struct SELU:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return 1.0507009873554804934193349852946 * (
                x * (x > 0).cast[T]()
                + 1.6732632423543772848170429916717 * (exp(x) - 1) * (x <= 0).cast[T]()
            )

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return 1.0507009873554804934193349852946 * (
                (x > 0).cast[T]()
                + 1.6732632423543772848170429916717 * exp(x) * (x <= 0).cast[T]()
            )

        return map[T](x, func)


struct Softplus:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return log(exp(x) + 1)

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Sigmoid.forward[T](x)


struct Softsign:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            return x / (abs(x) + 1)

        return map[T](x, func)

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1]) -> SIMD[T, 1]:
            let a = abs(x) + 1
            return 1 / (a * a)

        return map[T](x, func)
