from memory import memset
from random import rand, randint, randn
from math import sqrt


@value
struct Initializer:
    var name: String

    fn __init__(inout self, name: String):
        self.name = name

    fn initialize[T: DType](self, x: Tensor[T]) -> Tensor[T]:
        if self.name == "random_uniform":
            return RandomUniform.initialize(x)
        elif self.name == "xavier_normal":
            return XavierNormal.initialize(x)
        return Zero.initialize(x)

    fn initialize[T: DType](self, x: Tensor[T], value: Int) -> Tensor[T]:
        return Constant.initialize(x, value)


@value
struct Constant:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T], val: Int) -> Tensor[T]:
        let newData = Tensor[T](x.shape())
        for i in range(x.num_elements()):
            newData.data()[i] = val
        return newData


@value
struct Zero:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        let newData = Tensor[T](x.shape())
        for i in range(x.num_elements()):
            newData.data()[i] = 0
        return newData


@value
struct RandomUniform:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        rand[T](x.data(), x.num_elements())
        return x


@value
struct XavierNormal:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        let fan_in = x.shape()[0]
        let fan_out = x.shape()[1]
        let variance = 2.0 / (fan_in + fan_out)
        randn[T](x.data(), 0, variance)
        return x
