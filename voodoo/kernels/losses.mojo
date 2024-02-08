from math import log, abs

from voodoo.constants import EPSILON
from voodoo.kernels.generics import GenericLoss


trait Loss:
    ...


@register_passable("trivial")
struct MSE[](Loss):
    alias fw = GenericLoss[mse_error, mse_grad].fw
    alias bw = GenericLoss[mse_error, mse_grad].bw


@register_passable("trivial")
struct MAE[](Loss):
    alias fw = GenericLoss[mae_error, mae_grad].fw
    alias bw = GenericLoss[mae_error, mae_grad].bw


@register_passable("trivial")
struct MAPE[](Loss):
    alias fw = GenericLoss[mape_error, mape_grad].fw
    alias bw = GenericLoss[mape_error, mape_grad].bw


@register_passable("trivial")
struct MSLE[](Loss):
    alias fw = GenericLoss[msle_error, msle_grad].fw
    alias bw = GenericLoss[msle_error, msle_grad].bw


fn mse_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = (x - y)^2
    return (y_pred - y_true) ** 2.0


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


fn mae_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = |x - y|
    return abs(y_pred - y_true)


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


fn mape_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = |x - y| / y
    return abs(y_pred - y_true) / (y_true + EPSILON)


fn mape_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).select[DType.float32](-1.0, 1.0)


fn msle_error[
    NELTS: Int
](y_pred: SIMD[DType.float32, NELTS], y_true: SIMD[DType.float32, NELTS]) -> SIMD[
    DType.float32, NELTS
]:
    # f(x, y) = (log(x + 1) - log(y + 1))^2
    let y_pred_clipped = (y_pred > 0.0).select[DType.float32](y_pred, 0.0)
    let y_true_clipped = (y_true > 0.0).select[DType.float32](y_true, 0.0)
    return (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))) * (
        log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))
    )


fn msle_grad[
    NELTS: Int
](
    y_pred: SIMD[DType.float32, NELTS],
    y_true: SIMD[DType.float32, NELTS],
    cap: Float32,
    N: Int,
) -> SIMD[DType.float32, NELTS]:
    # f'(x, y) with respect to y = -2(log(x + 1) - log(y + 1)) / (y + 1)
    let y_pred_clipped = (y_pred > 0.0).select[DType.float32](y_pred, 0.0)
    let y_true_clipped = (y_true > 0.0).select[DType.float32](y_true, 0.0)
    return (
        -Float32(2.0)
        * (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0)))
        / (y_true_clipped + Float32(1.0))
    )
