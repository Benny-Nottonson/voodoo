from algorithm import vectorize
from math import exp, max, tanh


@value
struct Activation:
    var name: String

    fn __init__(inout self, name: String) -> None:
        self.name = name

    fn forward[T: DType](inout self, x: Tensor[T]) -> Tensor[T]:
        if self.name == "relu":
            return ReLU.forward(x)
        elif self.name == "tanh":
            return Tanh.forward(x)
        elif self.name == "sigmoid":
            return Sigmoid.forward(x)
        return Linear.forward(x)

    fn deriv[T: DType](inout self, x: Tensor[T]) -> Tensor[T]:
        if self.name == "relu":
            return ReLU.deriv(x)
        elif self.name == "tanh":
            return Tanh.deriv(x)
        elif self.name == "sigmoid":
            return Sigmoid.deriv(x)
        return Linear.deriv(x)

@value
struct Linear:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Tensor[T](x.shape(), 1)


@value
struct ReLU:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = Tensor[T](x.shape())

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            t_new.simd_store(idx, max(0, x.simd_load[simdwidthof[T]()](idx)))

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = Tensor[T](x.shape())

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            t_new.simd_store(
                idx, SIMD[T, 1](1) if x.simd_load[simdwidthof[T]()](idx) > 0 else 0
            )

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new


@value
struct Tanh:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = Tensor[T](x.shape())

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            t_new.simd_store(idx, tanh(x.simd_load[simdwidthof[T]()](idx)))

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = Tensor[T](x.shape())

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            t_new.simd_store(idx, 1 - tanh(x.simd_load[simdwidthof[T]()](idx)) ** 2)

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new


@value
struct Sigmoid:
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = x.clip(-500, 500)

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            t_new.simd_store(idx, 1 / (1 + exp(-x.simd_load[simdwidthof[T]()](idx))))

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = x.clip(-500, 500)

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            let f = 1 / (1 + exp(-x.simd_load[simdwidthof[T]()](idx)))
            t_new.simd_store(
                idx,
                f * (1 - f),
            )

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new
