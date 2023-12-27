from algorithm import vectorize
from math import exp, max, tanh
from utilities import map

@value
struct Activation[T: DType, name: String]:
    var forward: fn(Tensor[T]) -> Tensor[T]
    var deriv: fn(Tensor[T]) -> Tensor[T]

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
        else:
            raise Error("Unknown activation function: " + name)


@value
struct Linear:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Tensor[T](x.shape(), 1)


@value
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


@value
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


@value
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
