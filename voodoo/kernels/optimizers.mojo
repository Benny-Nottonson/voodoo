from .generics import GenericOptimizer


trait Optimizer:
    ...


@register_passable("trivial")
struct SGD[learning_rate: Float32](Optimizer):
    alias step = GenericOptimizer[sgd_step].step[learning_rate]


fn sgd_step[
    NELTS: Int, learning_rate: Float32
](grad: SIMD[DType.float32, NELTS]) -> SIMD[DType.float32, NELTS]:
    return grad * learning_rate
