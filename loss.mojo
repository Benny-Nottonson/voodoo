from algorithm import vectorize

trait LossFunction:
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Loss[T]:
        ...

struct Loss[T: DType]:
    var value: SIMD[DType.float64, 1]
    var delta: Tensor[T]

    fn __init__(inout self, value: SIMD[DType.float64, 1], delta: Tensor[T]):
        self.value = value
        self.delta = delta

@value
struct MeanSquaredError(LossFunction):
    @staticmethod
    fn calculate[T: DType](y_true: Tensor[T], y_pred: Tensor[T]) raises -> Loss[T]:
        var sum = SIMD[DType.float64, 1](0.0)

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            sum += ((y_true[idx] - y_pred[idx]) ** 2).cast[DType.float64]()

        vectorize[simdwidthof[T](), vecmath](y_true.num_elements())
        return Loss[T](sum / y_true.num_elements(), y_true - y_pred)

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