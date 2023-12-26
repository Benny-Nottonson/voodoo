from utilities import Matrix
from math import sqrt

trait LossFunction:
    ...

trait ProbabilisticLoss(LossFunction):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> Matrix:
        ...

trait RegressionLoss(LossFunction):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> SIMD[DType.float32, 1]:
        ...

struct CategoricalCrossentropy(ProbabilisticLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> Matrix:
        let SingleMatrix = Matrix(y_true.shape, 1)
        return -y_true * Matrix.log(y_pred) - (SingleMatrix - y_true) * Matrix.log(SingleMatrix - y_pred)

struct Possion(ProbabilisticLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> Matrix:
        return y_pred - y_true * Matrix.log(y_pred)

struct KLDivergence(ProbabilisticLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> Matrix:
        return y_true * Matrix.log(y_true / y_pred)

struct MeanSquaredError(RegressionLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> SIMD[DType.float32, 1]:
        return Matrix.mean(Matrix.square(y_pred - y_true))
    
struct MeanAbsoluteError(RegressionLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> SIMD[DType.float32, 1]:
        return Matrix.mean(Matrix.abs(y_pred - y_true))

struct MeanAbsolutePercentageError(RegressionLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> SIMD[DType.float32, 1]:
        return Matrix.mean(Matrix.abs(y_pred - y_true) / y_true)

struct MeanSquaredLogarithmicError(RegressionLoss):
    fn __call__(inout self, y_true: Matrix, y_pred: Matrix, sample_weight: Matrix) -> SIMD[DType.float32, 1]:
        return Matrix.mean(Matrix.square(Matrix.log(y_pred + 1) - Matrix.log(y_true + 1)))