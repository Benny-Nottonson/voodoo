from random import seed, rand
from algorithm import vectorize


fn random_uniform(
    data: DTypePointer[DType.float32], size: Int, other_params: StaticIntTuple[2]
) -> None:
    let low = other_params[0]
    let high = other_params[1]

    rand(data, size)

    @parameter
    fn vec_op[NELTS: Int](x: Int):
        data.simd_store[NELTS](x, data.simd_load[NELTS](x).fma((high - low), low))
