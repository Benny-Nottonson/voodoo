from algorithm import vectorize
from math import sqrt, max, abs
from tensor import TensorShape

from voodoo.utils import reduce_vector_mul
from voodoo.constants import NELTS, EPSILON


trait Constraint(CollectionElement):
    fn __init__(inout self):
        ...

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]) -> None:
        ...

    @staticmethod
    fn key() -> String:
        ...


@register_passable("trivial")
struct MaxNorm[max_value: Float32](CollectionElement, Constraint):
    """
    A constraint that enforces the maximum norm of the weights.
    """

    fn __init__() -> Self:
        return MaxNorm[max_value] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        let num_elements = reduce_vector_mul[shape]()
        var norms: Float32 = 0.0

        @parameter
        fn vec_norm[NELTS: Int](x: Int):
            norms += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[NELTS, vec_norm](num_elements)
        norms = sqrt(norms)
        let scale = max_value / (norms + EPSILON)

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, data.simd_load[NELTS](x) * scale)

        vectorize[NELTS, vec](num_elements)

    @staticmethod
    fn key() -> String:
        return "MaxNorm"


@register_passable("trivial")
struct MinMaxNorm[min_value: Float32, max_value: Float32](
    CollectionElement, Constraint
):
    """
    A constraint that enforces the minimum and maximum norm of the weights.
    """

    fn __init__() -> Self:
        return MinMaxNorm[min_value, max_value] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        let num_elements = reduce_vector_mul[shape]()
        var norms: Float32 = 0.0

        @parameter
        fn vec_norm[NELTS: Int](x: Int):
            norms += (data.simd_load[NELTS](x) ** 2).reduce_add()

        vectorize[NELTS, vec_norm](num_elements)
        norms = sqrt(norms)
        let scaleMax = max_value / (norms + EPSILON)
        let scaleMin = min_value / (norms + EPSILON)

        @parameter
        fn vec[NELTS: Int](x: Int):
            let d = data.simd_load[NELTS](x)
            let norm = d * (
                scaleMax if d > max_value else scaleMin if d < min_value else 1.0
            )
            data.simd_store[NELTS](x, norm)

        vectorize[NELTS, vec](num_elements)

    @staticmethod
    fn key() -> String:
        return "MinMaxNorm"


@register_passable("trivial")
struct NonNeg[](CollectionElement, Constraint):
    """
    A constraint that enforces non-negative weights.
    """

    fn __init__() -> Self:
        return NonNeg[] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        let num_elements = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, abs(data.simd_load[NELTS](x)))

        vectorize[NELTS, vec](num_elements)

    @staticmethod
    fn key() -> String:
        return "NonNeg"


@register_passable("trivial")
struct RadialConstraint[](CollectionElement, Constraint):
    """
    A constraint that enforces the radial constraint on the weights.
    """

    fn __init__() -> Self:
        return RadialConstraint[] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        let num_elements = reduce_vector_mul[shape]()
        let center = shape[0] // 2

        @parameter
        fn vec[NELTS: Int](x: Int):
            let i = x // shape[1]
            let j = x % shape[1]
            let d = sqrt((i - center) ** 2 + (j - center) ** 2)
            data.simd_store[NELTS](
                x, data.simd_load[NELTS](x) * (1.0 if d <= center else 0.0)
            )

        vectorize[NELTS, vec](num_elements)

    @staticmethod
    fn key() -> String:
        return "RadialConstraint"


@register_passable("trivial")
struct UnitNorm[](CollectionElement, Constraint):
    """
    A constraint that enforces the unit norm of the weights.
    """

    fn __init__() -> Self:
        return UnitNorm[] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        let num_elements = reduce_vector_mul[shape]()

        @parameter
        fn vec[NELTS: Int](x: Int):
            let norm = sqrt((data.simd_load[NELTS](x) ** 2).reduce_add())
            data.simd_store[NELTS](x, data.simd_load[NELTS](x) / (norm + EPSILON))

        vectorize[NELTS, vec](num_elements)

    @staticmethod
    fn key() -> String:
        return "UnitNorm"


@register_passable("trivial")
struct NoneConstraint[](CollectionElement, Constraint):
    """
    An constraint that does nothing.
    """

    fn __init__() -> Self:
        return NoneConstraint[] {}

    fn constrain[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        ...

    @staticmethod
    fn key() -> String:
        return "NoneConstraint"
