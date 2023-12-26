from random import random_float64, randn_float64
from math import sqrt
from utilities import Matrix


trait Initializer(Copyable):
    fn __call__(self, x: Matrix) -> Matrix:
        ...

    fn initialize(self, x: Matrix) -> Matrix:
        ...


@value
struct Constant(Initializer):
    var _c: Float32

    fn __call__(self, x: Matrix) -> Matrix:
        return self.initialize(x)

    fn initialize(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = self._c
        return temp


@value
struct RandomUniform(Initializer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.initialize(x)

    fn initialize(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = random_float64(0, 1).cast[DType.float32]()
        return temp


@value
struct XavierUniform(Initializer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.initialize(x)

    fn initialize(self, x: Matrix) -> Matrix:
        let fan_in = x.rows
        let fan_out = x.cols
        let std = math.sqrt(2 / (fan_in + fan_out))
        let a = std * math.sqrt(3)

        var temp = Matrix(fan_in, fan_out)
        for i in range(fan_in):
            for j in range(fan_out):
                temp[i, j] = random_float64(-a, a).cast[DType.float32]()
        return temp


@value
struct HeNormal(Initializer):
    var mode: String

    fn __init__(inout self, mode: String):
        self.mode = mode

    fn __call__(self, x: Matrix) -> Matrix:
        return self.initialize(x)

    fn initialize(self, x: Matrix) -> Matrix:
        let fan_in = x.rows
        let fan_out = x.cols
        let fan = fan_in if self.mode == String("fan_in") else fan_out
        let gain = sqrt(2)
        let std = gain / sqrt(fan)

        var temp = Matrix(fan_in, fan_out)
        for i in range(fan_in):
            for j in range(fan_out):
                temp[i, j] = randn_float64(0, std).cast[DType.float32]()
        return temp
