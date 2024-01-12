from math import abs, log, max
from algorithm import vectorize
from voodoo import Node
from ..constants import DType_F32, nelts, epsilon

# TODO: Rewrite to use generic functions where possible


alias generic_vectorized_fw = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]

alias generic_vectorized_fw_modifier = fn[nelts: Int] (Float32, Int, Int) -> Float32

alias generic_vectorized_fw_a = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], Float32, Int
) -> SIMD[DType_F32, nelts]

alias generic_vectorized_fw_b = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], Float32, Int
) -> SIMD[DType_F32, nelts]


struct Generic[
    fw_vec: generic_vectorized_fw,
    bw_vec_a: generic_vectorized_fw_a,
    bw_vec_b: generic_vectorized_fw_b,
]:
    @staticmethod
    fn fw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = y_pred.shape_ptr.load().len.load()
        let N = y_pred.shape_ptr.load().load(num_dims - 1)
        let cap = Float32(y_pred.load_cap())
        var e: Float32 = 0.0

        @parameter
        fn vectorized_fw[nelts: Int](i: Int):
            let error = fw_vec[nelts](
                y_true.load_data[nelts](i), y_pred.load_data[nelts](i)
            )
            e += error.reduce_add()

        vectorize[nelts, vectorized_fw](cap.to_int())
        node.store_data(0, e / cap / Float32(N))

    @staticmethod
    fn bw(node: Node, y_pred: Node, y_true: Node):
        let num_dims = y_pred.shape_ptr.load().len.load()
        let N = y_pred.shape_ptr.load().load(num_dims - 1)
        let cap = y_pred.load_cap()
        let scalar = cap / Float32(N)

        @parameter
        fn vectorized_mae_bw[nelts: Int](i: Int):
            let grad_a = bw_vec_a[nelts](
                y_true.load_data[nelts](i), y_pred.load_data[nelts](i), cap, N
            )
            let grad_b = bw_vec_b[nelts](
                grad_a, y_true.load_data[nelts](i), y_pred.load_data[nelts](i), cap, N
            )
            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad_a / scalar)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) + grad_b / scalar)

        vectorize[nelts, vectorized_mae_bw](y_pred.load_cap())


alias MSE = Generic[mse_error, mse_grad_a, mse_grad_b]
alias MAE = Generic[mae_error, mae_grad_a, mae_grad_b]
alias MAPE = Generic[mape_error, mape_grad_a, mape_grad_b]
alias MSLE = Generic[msle_error, msle_grad_a, msle_grad_b]
# alias CE = Generic[ce_error, ce_grad_a, ce_grad_b]


@parameter
fn mse_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = (x - y)^2
    return (y_pred - y_true) * (y_pred - y_true)


@parameter
fn mse_grad_a[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -2(x - y)
    return -Float32(2.0) * (y_pred - y_true)


@parameter
fn mse_grad_b[
    nelts: Int
](
    grad_a: SIMD[DType_F32, nelts],
    y_pred: SIMD[DType_F32, nelts],
    y_true: SIMD[DType_F32, nelts],
    cap: Float32,
    N: Int,
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 2(x - y)
    return -grad_a


@parameter
fn mae_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y|
    return abs(y_pred - y_true)


@parameter
fn mae_grad_a[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).cast[DType_F32]() * Float32(-2.0) + Float32(1.0)


@parameter
fn mae_grad_b[
    nelts: Int
](
    grad_a: SIMD[DType_F32, nelts],
    y_pred: SIMD[DType_F32, nelts],
    y_true: SIMD[DType_F32, nelts],
    cap: Float32,
    N: Int,
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 1 if x > y else -1
    return -grad_a


@parameter
fn mape_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y| / y
    return abs(y_pred - y_true) / (y_true + epsilon)


@parameter
fn mape_grad_a[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).cast[DType_F32]() * Float32(-2.0) + Float32(1.0)


@parameter
fn mape_grad_b[
    nelts: Int
](
    grad_a: SIMD[DType_F32, nelts],
    y_pred: SIMD[DType_F32, nelts],
    y_true: SIMD[DType_F32, nelts],
    cap: Float32,
    N: Int,
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 1 if x > y else -1
    return -grad_a


@parameter
fn msle_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = (log(x + 1) - log(y + 1))^2
    let y_pred_clipped = (y_pred > 0.0).cast[DType_F32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType_F32]() * y_true
    return (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))) * (
        log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0))
    )


@parameter
fn msle_grad_a[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -2(log(x + 1) - log(y + 1)) / (y + 1)
    let y_pred_clipped = (y_pred > 0.0).cast[DType_F32]() * y_pred
    let y_true_clipped = (y_true > 0.0).cast[DType_F32]() * y_true
    return (
        -Float32(2.0)
        * (log(y_pred_clipped + Float32(1.0)) - log(y_true_clipped + Float32(1.0)))
        / (y_true_clipped + Float32(1.0))
    )


@parameter
fn msle_grad_b[
    nelts: Int
](
    grad_a: SIMD[DType_F32, nelts],
    y_pred: SIMD[DType_F32, nelts],
    y_true: SIMD[DType_F32, nelts],
    cap: Float32,
    N: Int,
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to x = 2(log(x + 1) - log(y + 1)) / (x + 1)
    return -grad_a

'''
@parameter
fn ce_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    return -y_pred * log(y_true + epsilon)


@parameter
fn ce_grad_a[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    return -log(y_true + epsilon) / cap * Float32(N)


@parameter
fn ce_grad_b[
    nelts: Int
](
    grad_a: SIMD[DType_F32, nelts],
    y_pred: SIMD[DType_F32, nelts],
    y_true: SIMD[DType_F32, nelts],
    cap: Float32,
    N: Int,
) -> SIMD[DType_F32, nelts]:
    return -y_pred / (y_true + epsilon) / cap * Float32(N)
'''