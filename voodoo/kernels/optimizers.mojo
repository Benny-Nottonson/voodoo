from .generics import GenericOptimizer


struct SGD[learning_rate: Float32]:
    alias step = GenericOptimizer[sgd_step].step[learning_rate]


@always_inline
fn sgd_step[
    nelts: Int, learning_rate: Float32
](grad: SIMD[DType.float32, nelts]) -> SIMD[DType.float32, nelts]:
    return grad * learning_rate
