from algorithm import vectorize
from random import (
    seed,
    random_float64,
    randn_float64,
    randn,
)
from .constants import NELTS
from .utils import reduce_vector_mul
from tensor import TensorShape


trait Initializer(CollectionElement):
    fn __init__(inout self):
        ...

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]) -> None:
        ...

    @staticmethod
    fn key() -> String:
        ...


@register_passable("trivial")
struct Constant[value: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with a constant value.
    """

    fn __init__() -> Self:
        return Constant[value] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, value.to_int())

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "Constant"


@register_passable("trivial")
struct Zeroes[](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with zeros.
    """

    fn __init__() -> Self:
        return Zeroes[] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, 0.0)

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "Zeroes"


@register_passable("trivial")
struct Ones[](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with ones.
    """

    fn __init__() -> Self:
        return Ones[] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, 1.0)

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "Ones"


@register_passable("trivial")
struct GlorotNormal[input_units: Float64, output_units: Float64](CollectionElement):
    """
    An initializer that fills a Tensor with values from a Glorot normal distribution, also known as Xavier normal distribution.
    """

    fn __init__() -> Self:
        return GlorotNormal[input_units, output_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        randn(
            data,
            reduce_vector_mul[shape](),
            0.0,
            (2.0 / (input_units + output_units)) ** 0.5,
        )

    @staticmethod
    fn key() -> String:
        return "GlorotNormal"


@register_passable("trivial")
struct GlorotUniform[input_units: Float64, output_units: Float64](CollectionElement):
    """
    An initializer that fills a Tensor with values from a Glorot uniform distribution, also known as Xavier uniform distribution.
    """

    fn __init__() -> Self:
        return GlorotUniform[input_units, output_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        let limit = (6.0 / (input_units + output_units)) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "GlorotUniform"


@register_passable("trivial")
struct HeNormal[input_units: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a He normal distribution.
    """

    fn __init__() -> Self:
        return HeNormal[input_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), 0.0, (2.0 / input_units) ** 0.5)

    @staticmethod
    fn key() -> String:
        return "HeNormal"


@register_passable("trivial")
struct HeUniform[input_units: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a He uniform distribution.
    """

    fn __init__() -> Self:
        return HeUniform[input_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        let limit = (6.0 / input_units) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "HeUniform"


@register_passable("trivial")
struct Identity[](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with the identity matrix. Must be a 2D tensor.
    """

    fn __init__() -> Self:
        return Identity[] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        let n = shape[0]
        let m = shape[1]

        @parameter
        fn vec[NELTS: Int](x: Int):
            let i = x / m
            let j = x % m
            data.simd_store[NELTS](x, 1.0 if i == j else 0.0)

        vectorize[NELTS, vec](n * m)

    @staticmethod
    fn key() -> String:
        return "Identity"


@register_passable("trivial")
struct LecunNormal[input_units: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a Lecun normal distribution.
    """

    fn __init__() -> Self:
        return LecunNormal[input_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), 0.0, (1.0 / input_units) ** 0.5)

    @staticmethod
    fn key() -> String:
        return "LecunNormal"


@register_passable("trivial")
struct LecunUniform[input_units: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a Lecun uniform distribution.
    """

    fn __init__() -> Self:
        return LecunUniform[input_units] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        let limit = (3.0 / input_units) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "LecunUniform"


@register_passable("trivial")
struct RandomNormal[mean: Float64, std: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a normal distribution.
    """

    fn __init__() -> Self:
        return RandomNormal[mean, std] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), mean, std)

    @staticmethod
    fn key() -> String:
        return "RandomNormal"


@register_passable("trivial")
struct RandomUniform[low: Float64, high: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a uniform distribution.
    """

    fn __init__() -> Self:
        return RandomUniform[low, high] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, random_float64(low, high).cast[DType.float32]())

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "RandomUniform"


@register_passable("trivial")
struct TruncatedNormal[mean: Float64, std: Float64](CollectionElement, Initializer):
    """
    An initializer that fills a Tensor with values from a truncated normal distribution.
    """

    fn __init__() -> Self:
        return TruncatedNormal[mean, std] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        seed()
        let low = mean - 2.0 * std
        let high = mean + 2.0 * std

        @parameter
        fn vec[NELTS: Int](x: Int):
            var value = randn_float64(mean, std)
            while value < low or value > high:
                value = randn_float64(mean, std)
            data.simd_store[NELTS](x, value.cast[DType.float32]())

        vectorize[NELTS, vec](reduce_vector_mul[shape]())

    @staticmethod
    fn key() -> String:
        return "TruncatedNormal"


@register_passable("trivial")
struct NoneInitializer[](CollectionElement, Initializer):
    """
    An initializer that does nothing.
    """

    fn __init__() -> Self:
        return NoneInitializer[] {}

    fn initialize[shape: Vector[Int]](self, data: DTypePointer[DType.float32]):
        ...

    @staticmethod
    fn key() -> String:
        return "NoneInitializer"
