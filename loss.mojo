from algorithm import vectorize
from utilities import reduce


@value
struct Loss[T: DType]:
    var value: SIMD[T, 1]
    var delta: Tensor[T]

    fn __init__(inout self, value: SIMD[T, 1], delta: Tensor[T]):
        self.value = value
        self.delta = delta

@value
struct LossFunction[T: DType, name: String]:
    var pred: Tensor[T]
    var true: Tensor[T]
    var epsilon: Float64
    var _calculate: fn(Tensor[T], Tensor[T]) raises -> Loss[T]

    fn __init__(inout self) raises:
        self.pred = Tensor[T]()
        self.true = Tensor[T]()
        self.epsilon = 1e-7
        if name == "mse":
            self._calculate = MeanSquaredError.calculate[T]
        else:
            raise Error("Unknown loss function: " + name)

    fn calculate(inout self, pred: Tensor[T], true: Tensor[T]) raises -> Loss[T]:
        self.pred = pred
        self.true = true
        return self._calculate(pred, true)

@value
struct MeanSquaredError:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Loss[T]:
        @noncapturing
        fn func(x: SIMD[T, 1], y: SIMD[T, 1]) -> SIMD[T, 1]:
            return (x - y) * (x - y)

        return Loss(reduce[T](y_true, y_pred, func), y_true - y_pred)


"""
struct CategoricalCrossentropy(LossFunction):
    ...

struct Possion(LossFunction):
    ...

struct KLDivergence(LossFunction):
    ...
    
struct MeanAbsoluteError(LossFunction):
    ...

struct MeanAbsolutePercentageError(LossFunction):
    ...

struct MeanSquaredLogarithmicError(LossFunction):
    ...
"""
