from memory import memset
from random import rand, randint, randn
from math import sqrt


@value
struct Initializer[T: DType, name: String]:
    var initialize: fn(Tensor[T]) -> None

    fn __init__(inout self) raises:
        if name == "random_uniform":
            self.initialize = RandomUniform.initialize[T]
        elif name == "xavier_normal":
            self.initialize = XavierNormal.initialize[T]
        elif name == "zeros":
            self.initialize = Zeros.initialize[T]
        else:
            raise Error("Unknown initializer: " + name)

@value
struct Zeros:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]):
        memset(x.data(), 0, x.num_elements())

@value
struct RandomUniform:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]):
        rand[T](x.data(), x.num_elements())


@value
struct XavierNormal:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]):
        let fan_in = x.shape()[0]
        let fan_out = x.shape()[1]
        let variance = 2.0 / (fan_in + fan_out)
        randn[T](x.data(), 0, variance)
