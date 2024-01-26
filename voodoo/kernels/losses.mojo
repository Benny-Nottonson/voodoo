from .generics import GenericLoss
from math import log, abs
from ..constants import EPSILON


struct MSE[]:
    alias fw = GenericLoss[mse_error, mse_grad].fw
    alias bw = GenericLoss[mse_error, mse_grad].bw


struct MAE[]:
    alias fw = GenericLoss[mae_error, mae_grad].fw
    alias bw = GenericLoss[mae_error, mae_grad].bw


struct MAPE[]:
    alias fw = GenericLoss[mape_error, mape_grad].fw
    alias bw = GenericLoss[mape_error, mape_grad].bw


struct MSLE[]:
    alias fw = GenericLoss[msle_error, msle_grad].fw
    alias bw = GenericLoss[msle_error, msle_grad].bw


@always_inline
fn mse_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = (x - y)^2
    return (y_pred - y_true) ** 2.0


@always_inline
fn mse_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -2(x - y)
    return -2.0 * (y_pred - y_true)


@always_inline
fn mae_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = |x - y|
    return abs(y_pred - y_true)


@always_inline
fn mae_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).select(Float32(-1.0), 1.0)


@always_inline
fn mape_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = |x - y| / y
    return abs(y_pred - y_true) / (y_true + EPSILON)


@always_inline
fn mape_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).cast[DType.float32]() * Float32(-2.0) + Float32(1.0)


@always_inline
fn msle_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = (log(x + 1) - log(y + 1))^2
    let y_pred_clipped = (y_pred > 0.0).cast[DType.float32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType.float32]() * y_true
    return (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))) * (
        log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))
    )


@always_inline
fn msle_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -2(log(x + 1) - log(y + 1)) / (y + 1)
    let y_pred_clipped = (y_pred > 0.0).cast[DType.float32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType.float32]() * y_true
    return (
        -Float32(2.0)
        * (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0)))
        / (y_true_clipped + Float32(1.0))
    )
