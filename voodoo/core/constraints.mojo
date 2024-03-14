from algorithm import vectorize
from math import sqrt, max, abs
from tensor import TensorShape

from voodoo.utils import Vector, reduce_vector_mul
from voodoo.constants import NELTS, EPSILON


trait Constraint(CollectionElement):
    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]) -> None:
        ...


@register_passable("trivial")
struct MaxNorm[max_value: Float32](Constraint):
    """
    A constraint that enforces the maximum norm of the weights.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        var num_elements = reduce_vector_mul[shape]()
        var norms: Float32 = 0.0

        @parameter
        fn vec_norm[NELTS: Int](x: Int):
            norms += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[vec_norm, NELTS](num_elements)
        norms = sqrt(norms)
        var scale = max_value / (norms + EPSILON)

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, data.simd_load[NELTS](x) * scale)

        vectorize[vec, NELTS](num_elements)


@register_passable("trivial")
struct MinMaxNorm[min_value: Float32, max_value: Float32](Constraint):
    """
    A constraint that enforces the minimum and maximum norm of the weights.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        var num_elements = reduce_vector_mul[shape]()
        var norms: Float32 = 0.0

        @parameter
        fn vec_norm[NELTS: Int](x: Int):
            norms += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[vec_norm, NELTS](num_elements)
        norms = sqrt(norms)
        var scaleMax = max_value / (norms + EPSILON)
        var scaleMin = min_value / (norms + EPSILON)

        @parameter
        fn vec[NELTS: Int](x: Int):
            var d = data.simd_load[NELTS](x)
            var norm = d * (
                scaleMax if d > max_value else scaleMin if d < min_value else 1.0
            )
            data.simd_store[NELTS](x, norm)

        vectorize[vec, NELTS](num_elements)


@register_passable("trivial")
struct NonNeg[](Constraint):
    """
    A constraint that enforces non-negative weights.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        var num_elements = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, abs(data.simd_load[NELTS](x)))

        vectorize[vec, NELTS](num_elements)


@register_passable("trivial")
struct RadialConstraint[](Constraint):
    """
    A constraint that enforces the radial constraint on the weights.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        var num_elements = reduce_vector_mul[shape]()
        var center = shape[0] // 2

        @parameter
        fn vec[NELTS: Int](x: Int):
            var i = x // shape[1]
            var j = x % shape[1]
            var d = sqrt((i - center) ** 2 + (j - center) ** 2)
            data.simd_store[NELTS](
                x, data.simd_load[NELTS](x) * (1.0 if d <= center else 0.0)
            )

        vectorize[vec, NELTS](num_elements)


@register_passable("trivial")
struct UnitNorm[](Constraint):
    """
    A constraint that enforces the unit norm of the weights.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        var num_elements = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            var norm = sqrt((data.simd_load[NELTS](x) ** 2).reduce_add())
            data.simd_store[NELTS](x, data.simd_load[NELTS](x) / (norm + EPSILON))

        vectorize[vec, NELTS](num_elements)


@register_passable("trivial")
struct NoneConstraint[](Constraint):
    """
    An constraint that does nothing.
    """

    @staticmethod
    fn constrain[shape: Vector[Int]](data: DTypePointer[DType.float32]):
        ...
