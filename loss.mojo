from algorithm import vectorize


@value
struct Loss[T: DType]:
    var value: SIMD[T, 1]
    var delta: Tensor[T]

    fn __init__(inout self, value: SIMD[T, 1], delta: Tensor[T]):
        self.value = value
        self.delta = delta

@value
struct MeanSquaredError[T: DType]:
    var pred: Tensor[T]
    var true: Tensor[T]
    var epsilon: Float64

    fn __init__(inout self, epsilon: Float64 = 1e-7):
        self.pred = Tensor[T]()
        self.true = Tensor[T]()
        self.epsilon = epsilon

    fn calculate(inout self, y_true: Tensor[T], y_pred: Tensor[T]) raises -> Loss[T]:
        self.pred = y_pred
        self.true = y_true
        var sum = SIMD[DType.float64, 1](0.0)

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            sum += ((y_true[idx] - y_pred[idx]) ** 2).cast[DType.float64]()

        vectorize[simdwidthof[T](), vecmath](y_true.num_elements())
        return Loss[T]((sum / y_true.num_elements()).cast[T](), y_true - y_pred)


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
