from algorithm import vectorize
from utilities import reduce, map, apply
from math import log, max, abs, cosh

alias lossFunctions = VariadicList[StringLiteral](
    "mse",
    "binary_crossentropy",
    "categorical_crossentropy",
    "cosine_similarity",
    "hinge",
    "huber",
    "kl_divergence",
    "log_cosh",
    "mae",
    "mape",
    "poisson",
    "squared_hinge",
)


struct Loss[T: DType, name: String]:
    var calculate: fn (Tensor[T], Tensor[T]) raises -> Tensor[T]

    fn __init__(inout self) raises:
        if name == "mse":
            self.calculate = MeanSquaredError.calculate[T]
        elif name == "binary_crossentropy":
            self.calculate = BinaryCrossentropy.calculate[T]
        elif name == "categorical_crossentropy":
            self.calculate = CategoricalCrossentropy.calculate[T]
        elif name == "cosine_similarity":
            self.calculate = CosineSimilarity.calculate[T]
        elif name == "hinge":
            self.calculate = Hinge.calculate[T]
        elif name == "huber":
            self.calculate = Huber.calculate[T]
        elif name == "kl_divergence":
            self.calculate = KLDivergence.calculate[T]
        elif name == "log_cosh":
            self.calculate = LogCosh.calculate[T]
        elif name == "mae":
            self.calculate = MeanAbsoluteError.calculate[T]
        elif name == "mape":
            self.calculate = MeanAbsolutePercentageError.calculate[T]
        elif name == "poisson":
            self.calculate = Poisson.calculate[T]
        elif name == "squared_hinge":
            self.calculate = SquaredHinge.calculate[T]
        else:
            raise Error("Invalid loss function name: " + name)


struct MeanSquaredError:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return (x - y) * (x - y)

        return reduce[T](func, y_true, y_pred, 0)


struct BinaryCrossentropy:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return -x * log(y) - (1 - x) * log(1 - y)

        return reduce[T](func, y_true, y_pred, 0)


struct CategoricalCrossentropy:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return -x * log(y)

        return reduce[T](func, y_true, y_pred, 0)


struct CosineSimilarity:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return x * y

        return reduce[T](func, y_true, y_pred, 0)


struct Hinge:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return max(0, 1 - x * y)

        return reduce[T](func, y_true, y_pred, 0)


struct Huber:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            if abs(x - y) <= 1:
                return 0.5 * (x - y) * (x - y)
            else:
                return abs(x - y) - 0.5

        return reduce[T](func, y_true, y_pred, 0)


struct KLDivergence:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return x * log(x / y)

        return reduce[T](func, y_true, y_pred, 0)


struct LogCosh:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return log(cosh(x - y))

        return reduce[T](func, y_true, y_pred, 0)


struct MeanAbsoluteError:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return abs(x - y)

        return reduce[T](func, y_true, y_pred, 0)


struct MeanAbsolutePercentageError:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return abs((x - 1e-7) - (y - 1e-7)) / x

        return reduce[T](func, y_true, y_pred, 0)


struct Poisson:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return y - x * log(y + 1e-7)

        return reduce[T](func, y_true, y_pred, 0)


struct SquaredHinge:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Tensor[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return max(0, 1 - x * y) * max(0, 1 - x * y)

        return reduce[T](func, y_true, y_pred, 0)
