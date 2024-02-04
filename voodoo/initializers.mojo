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
from math import min
from .constants import NELTS
from .utils import reduce_vector_mul


trait Initializer:
    @staticmethod
    fn initialize[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        ...


struct Constant(Initializer):
    """
    An initializer that fills a Tensor with a constant value.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        value: Float64,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, value.to_int())

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct Zeroes(Initializer):
    """
    An initializer that fills a Tensor with zeros.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        arg0: Float64 = 0.0,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, 0.0)

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct Ones(Initializer):
    """
    An initializer that fills a Tensor with ones.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        arg0: Float64 = 0.0,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, 1.0)

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct GlorotNormal(Initializer):
    """
    An initializer that fills a Tensor with values from a Glorot normal distribution, also known as Xavier normal distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        output_units: Float64,
    ](data: DTypePointer[DType.float32]):
        seed()
        randn(
            data,
            reduce_vector_mul[shape](),
            0.0,
            (2.0 / (input_units + output_units)) ** 0.5,
        )


struct GlorotUniform(Initializer):
    """
    An initializer that fills a Tensor with values from a Glorot uniform distribution, also known as Xavier uniform distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        output_units: Float64,
    ](data: DTypePointer[DType.float32]):
        seed()
        let limit = (6.0 / (input_units + output_units)) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct HeNormal(Initializer):
    """
    An initializer that fills a Tensor with values from a He normal distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), 0.0, (2.0 / input_units) ** 0.5)


struct HeUniform(Initializer):
    """
    An initializer that fills a Tensor with values from a He uniform distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()
        let limit = (6.0 / input_units) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct Identity(Initializer):
    """
    An initializer that fills a Tensor with the identity matrix. Must be a 2D tensor.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        arg0: Float64 = 0.0,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()
        let n = shape[0]
        let m = shape[1]

        @parameter
        fn vec[NELTS: Int](x: Int):
            let i = x / m
            let j = x % m
            data.simd_store[NELTS](x, 1.0 if i == j else 0.0)

        vectorize[NELTS, vec](n * m)


struct LecunNormal(Initializer):
    """
    An initializer that fills a Tensor with values from a Lecun normal distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), 0.0, (1.0 / input_units) ** 0.5)


struct LecunUniform(Initializer):
    """
    An initializer that fills a Tensor with values from a Lecun uniform distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        input_units: Float64,
        arg1: Float64 = 0.0,
    ](data: DTypePointer[DType.float32]):
        seed()
        let limit = (3.0 / input_units) ** 0.5

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](
                x, random_float64(-limit, limit).cast[DType.float32]()
            )

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct RandomNormal(Initializer):
    """
    An initializer that fills a Tensor with values from a normal distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        mean: Float64,
        std: Float64,
    ](data: DTypePointer[DType.float32]):
        seed()
        randn(data, reduce_vector_mul[shape](), mean, std)


struct RandomUniform(Initializer):
    """
    An initializer that fills a Tensor with values from a uniform distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        low: Float64,
        high: Float64,
    ](data: DTypePointer[DType.float32]):
        seed()

        @parameter
        fn vec[NELTS: Int](x: Int):
            data.simd_store[NELTS](x, random_float64(low, high).cast[DType.float32]())

        vectorize[NELTS, vec](reduce_vector_mul[shape]())


struct TruncatedNormal(Initializer):
    """
    An initializer that fills a Tensor with values from a truncated normal distribution.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        mean: Float64,
        std: Float64,
    ](data: DTypePointer[DType.float32]):
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


struct NoneInitializer(Initializer):
    """
    An initializer that does nothing.
    """

    @staticmethod
    @always_inline("nodebug")
    fn initialize[
        shape: Vector[Int],
        arg0: Float64,
        arg1: Float64,
    ](data: DTypePointer[DType.float32]):
        ...
