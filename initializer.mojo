from memory import memset
from random import rand
from math import sqrt


trait Initializer:
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        ...

    @staticmethod
    fn initialize[T: DType](x: Tensor[T], value: SIMD[T, 1]) -> Tensor[T]:
        ...


@value
struct RandomUniform(Initializer):
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        rand[T](x.data(), x.num_elements())
        return x

    @staticmethod
    fn initialize[T: DType](x: Tensor[T], value: SIMD[T, 1]) -> Tensor[T]:
        return Tensor[T](x.shape(), value)


@value
struct XavierNormal(Initializer):
    @staticmethod
    fn initialize[T: DType](x: Tensor[T]) -> Tensor[T]:
        let fan_in = x.shape()[0]
        let fan_out = x.shape()[1]
        let scale = 2.0 / (fan_in + fan_out)
        let std = sqrt(SIMD[T, 1](scale))
        rand[T](x.data(), x.num_elements())
        return x * std

    @staticmethod
    fn initialize[T: DType](x: Tensor[T], value: SIMD[T, 1]) -> Tensor[T]:
        return Tensor[T](x.shape(), value)
