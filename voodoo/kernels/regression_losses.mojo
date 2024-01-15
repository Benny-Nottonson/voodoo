from math import abs, log, max, sqrt
from algorithm import vectorize
from voodoo import Node


alias generic_vectorized_fw = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts]
) -> SIMD[DType_F32, nelts]

alias generic_vectorized_bw = fn[nelts: Int] (
    SIMD[DType_F32, nelts], SIMD[DType_F32, nelts], Float32, Int
) -> SIMD[DType_F32, nelts]


struct Generic[
    fw_vec: generic_vectorized_fw,
    bw_vec: generic_vectorized_bw,
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
            let grad = bw_vec[nelts](
                y_true.load_data[nelts](i), y_pred.load_data[nelts](i), cap, N
            ) / scalar

            y_pred.store_grad[nelts](i, y_pred.load_grad[nelts](i) + grad)
            y_true.store_grad[nelts](i, y_true.load_grad[nelts](i) - grad)

        vectorize[nelts, vectorized_mae_bw](y_pred.load_cap())


alias MSE = Generic[mse_error, mse_grad]
alias MAE = Generic[mae_error, mae_grad]
alias MAPE = Generic[mape_error, mape_grad]
alias MSLE = Generic[msle_error, msle_grad]


@parameter
@always_inline
fn mse_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = (x - y)^2
    return (y_pred - y_true) ** 2.0


@parameter
@always_inline
fn mse_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -2(x - y)
    return -2.0 * (y_pred - y_true)


@parameter
@always_inline
fn mae_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y|
    return abs(y_pred - y_true)


@parameter
@always_inline
fn mae_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).select(Float32(-1.0), 1.0)


@parameter
@always_inline
fn mape_error[
    nelts: Int
](y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts]) -> SIMD[
    DType_F32, nelts
]:
    # f(x, y) = |x - y| / y
    return abs(y_pred - y_true) / (y_true + epsilon)


@parameter
@always_inline
fn mape_grad[
    nelts: Int
](
    y_pred: SIMD[DType_F32, nelts], y_true: SIMD[DType_F32, nelts], cap: Float32, N: Int
) -> SIMD[DType_F32, nelts]:
    # f'(x, y) with respect to y = -1 if x > y else 1
    return (y_pred > y_true).cast[DType_F32]() * Float32(-2.0) + Float32(1.0)


@parameter
@always_inline
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
@always_inline
fn msle_grad[
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
