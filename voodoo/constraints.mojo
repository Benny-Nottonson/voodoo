from algorithm import vectorize
from random import (
    seed,
    random_float64,
    random_si64,
    randint,
    rand,
    randn_float64,
    randn,
)
from math import min, sqrt, max, abs
from .constants import NELTS, EPSILON
from .utils import reduce_vector_mul


trait Constraint:
    @staticmethod
    fn constrain[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        ...


struct MaxNorm(Constraint):
    """
    A constraint that enforces the maximum norm of the weights. Norm is the square root of the sum of the squared elements.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        max_norm: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        var norm: Float32 = 0.0
        var elems = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            norm += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[NELTS, vec](elems)

        norm = sqrt(norm)

        if norm > max_norm.cast[DType.float32]():
            var scale = max_norm.cast[DType.float32]() / (norm + EPSILON)

            @parameter
            fn vec_2[NELTS: Int](x: Int):
                data.simd_store[NELTS](x, data.simd_load[NELTS](x) * scale)

            vectorize[NELTS, vec_2](elems)


struct MinMaxNorm(Constraint):
    """
    A constraint that enforces the minimum and maximum norm of the weights. Norm is the square root of the sum of the squared elements.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        min_value: Float64,
        max_value: Float64,
    ](data: DTypePointer[DType.float32]):
        var elems = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            var norm: Float32 = (data.simd_load[NELTS](x) ** 2).reduce_add()
            norm = sqrt(norm)

            if norm < min_value.cast[DType.float32]():
                var scale = min_value.cast[DType.float32]() / (norm + EPSILON)
                data.simd_store[NELTS](x, data.simd_load[NELTS](x) * scale)
            elif norm > max_value.cast[DType.float32]():
                var scale = max_value.cast[DType.float32]() / (norm + EPSILON)
                data.simd_store[NELTS](x, data.simd_load[NELTS](x) * scale)

        vectorize[NELTS, vec](elems)


struct NonNeg(Constraint):
    """
    A constraint that enforces non-negative weights.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        var elems = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, abs(data.simd_load[NELTS](x)))

        vectorize[NELTS, vec](elems)


struct RadialConstraint(Constraint):
    """
    A constraint that enforces the radial constraint of the weights.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        var elems = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            var center = data.simd_load[NELTS](x)
            var corner = data.simd_load[NELTS](x + 3)
            var side = data.simd_load[NELTS](x + 1)
            var diag = data.simd_load[NELTS](x + 2)

            var avg = (center + corner + side + diag) / 4.0
            data.simd_store[NELTS](x, avg)
            data.simd_store[NELTS](x + 3, avg)
            data.simd_store[NELTS](x + 1, avg)
            data.simd_store[NELTS](x + 2, avg)

        vectorize[NELTS, vec](elems)


struct UnitNorm(Constraint):
    """
    A constraint that enforces the weights to have unit norm.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        var norm: Float32 = 0.0
        var elems = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            norm += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[NELTS, vec](elems)

        norm = sqrt(norm)

        @parameter
        fn vec_2[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, data.simd_load[NELTS](x) / (norm + EPSILON))

        vectorize[NELTS, vec_2](elems)


struct NoneConstraint(Constraint):
    """
    A constraint that does nothing.
    """

    @staticmethod
    @always_inline("nodebug")
    fn constrain[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        ...
