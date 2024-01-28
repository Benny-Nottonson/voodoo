from .generics import GenericOptimizer


struct SGD[learning_rate: Float32]:
    alias step = GenericOptimizer[sgd_step].step[learning_rate]


@always_inline("nodebug")
fn sgd_step[
    NELTS: Int, learning_rate: Float32
](grad: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return grad * learning_rate
