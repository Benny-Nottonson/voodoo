from memory import memset
from random import rand, randint, randn
from math import sqrt


@value
struct Regularizer[T: DType, name: String]:
    var regularize: fn(Tensor[T]) -> None

    fn __init__(inout self) raises:
        if name == "l1":
            self.regularize = L1.regularize[T]
        elif name == "l2":
            self.regularize = L2.regularize[T]
        else:
            raise Error("Unknown regularizer: " + name)

@value
struct L1:
    @staticmethod
    fn regularize[T: DType](x: Tensor[T]):
        ...

@value
struct L2:
    @staticmethod
    fn regularize[T: DType](x: Tensor[T]):
        ...