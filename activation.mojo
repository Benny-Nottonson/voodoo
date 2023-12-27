from layer import Layer
from algorithm import vectorize
from math import exp, max, tanh


@value
trait ActivationLayer(Layer):
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        ...

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        ...


@value
struct Linear(ActivationLayer):
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x

    @staticmethod
    fn deriv[T: DType](x: Tensor[T]) -> Tensor[T]:
        return Tensor[T](x.shape(), 1)


@value
struct ReLU(ActivationLayer):
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
struct Tanh(ActivationLayer):
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
struct Sigmoid[T: DType](ActivationLayer):
    @staticmethod
    fn forward[T: DType](x: Tensor[T]) -> Tensor[T]:
        var t_new = Tensor[T](x.shape())

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
            t_new.simd_store(idx, t_new.simd_load[simdwidthof[T]()](idx) * (1 - t_new.simd_load[simdwidthof[T]()](idx)))

        vectorize[simdwidthof[T](), vecmath](x.num_elements())
        return t_new
