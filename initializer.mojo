from random import rand, randn
from tensor import TensorShape
from math import sqrt, min

alias initializerFunctions = VariadicList[StringLiteral](
    "zeros",
    "ones",
    "glorot_normal",
    "glorot_uniform",
    "he_normal",
    "he_uniform",
    "identity",
    "lecun_normal",
    "lecun_uniform",
    "random_normal",
    "random_uniform",
    "truncated_normal",
)


@value
struct Initializer[T: DType, name: String]:
    var initialize: fn (TensorShape) -> Tensor[T]

    fn __init__(inout self) raises:
        if name == "zeros":
            self.initialize = Zeros.initialize[T]
        elif name == "ones":
            self.initialize = Ones.initialize[T]
        elif name == "glorot_normal":
            self.initialize = GlorotNormal.initialize[T]
        elif name == "glorot_uniform":
            self.initialize = GlorotUniform.initialize[T]
        elif name == "he_normal":
            self.initialize = HeNormal.initialize[T]
        elif name == "he_uniform":
            self.initialize = HeUniform.initialize[T]
        elif name == "identity":
            self.initialize = Identity.initialize[T]
        elif name == "lecun_normal":
            self.initialize = LecunNormal.initialize[T]
        elif name == "lecun_uniform":
            self.initialize = LecunUniform.initialize[T]
        elif name == "random_normal":
            self.initialize = RandomNormal.initialize[T]
        elif name == "random_uniform":
            self.initialize = RandomUniform.initialize[T]
        elif name == "truncated_normal":
            self.initialize = TruncatedNormal.initialize[T]
        else:
            raise Error("Unknown initializer: " + name)

    fn constant_initialize(self, x: TensorShape, v: SIMD[T, 1]) -> Tensor[T]:
        var t = Tensor[T](x)
        for i in range(t.num_elements()):
            t[i] = v
        return t


struct Zeros:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        var t = Tensor[T](x)
        for i in range(t.num_elements()):
            t[i] = 0
        return t


struct Ones:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        var t = Tensor[T](x)
        for i in range(t.num_elements()):
            t[i] = 1
        return t


# rand[T](tensorShape) -> Random uniform tensor
# randn[T](tensorShape, mean, variance) -> Random normal tensor


struct GlorotNormal:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        let fan_out = x[1]
        return randn[T](x, 0, 2.0 / (fan_in + fan_out))


struct GlorotUniform:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        let fan_out = x[1]
        let t = rand[T](x)
        let s: SIMD[T, 1] = 6.0 / (fan_in + fan_out)
        return t * sqrt(s)


struct HeNormal:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        return randn[T](x, 0, 2.0 / fan_in)


struct HeUniform:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        let t = rand[T](x)
        let s: SIMD[T, 1] = 6.0 / fan_in
        return t * sqrt(s)


struct Identity:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        var t = Tensor[T](x, 0)
        let n = min(x[0], x[1])
        for i in range(n):
            t[i][i] = 1
        return t


struct LecunNormal:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        return randn[T](x, 0, 1.0 / sqrt(fan_in))


struct LecunUniform:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        let fan_in = x[0]
        let t = rand[T](x)
        let s: SIMD[T, 1] = 3.0 / sqrt(fan_in)
        return t * sqrt(s)


struct RandomNormal:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        return randn[T](x, 0, 1)


struct RandomUniform:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        return rand[T](x)


struct TruncatedNormal:
    @staticmethod
    fn initialize[T: DType](x: TensorShape) -> Tensor[T]:
        return randn[T](x, 0, 1).clip(-2, 2)
