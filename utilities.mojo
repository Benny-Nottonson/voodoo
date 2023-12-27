from algorithm import vectorize


fn map[T: DType](x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> Tensor[T]:
    var t_new = Tensor[T](x.shape())

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        t_new[i][j] = f(x[i][j])

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return t_new


fn apply[T: DType](owned x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> None:
    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        x[i][j] = f(x[i][j])

    vectorize[simdwidthof[T](), vecmath](x.num_elements())


fn reduce[T: DType](x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> SIMD[T, 1]:
    var acc = SIMD[T, 1](0)

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        acc += f(x[i][j])

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return acc


fn reduce[
    T: DType
](
    f: fn (SIMD[T, 1], SIMD[T, 1]) -> SIMD[T, 1], x: Tensor[T], y: Tensor[T], axis: Int
) -> Tensor[T]:
    var targetAxis = axis
    if axis == -1:
        targetAxis = x.shape()[x.rank() - 1]

    var t_new = Tensor[T](x.shape()[1 - targetAxis])

    for i in range(0, x.shape()[1 - targetAxis]):
        var acc = SIMD[T, 1](0)

        @parameter
        fn vecmath[simd_width: Int](idx: Int) -> None:
            let j = idx % x.shape()[1]
            acc += f(x[i][j], y[i][j])

        vectorize[simdwidthof[T](), vecmath](x.shape()[targetAxis])
        t_new[i] = acc

    return t_new


fn transposition[T: DType](x: Tensor[T]) -> Tensor[T]:
    var t_new = Tensor[T](x.shape()[1], x.shape()[0])

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        t_new[j][i] = x[i][j]

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return t_new


fn axis_sum[T: DType](x: Tensor[T], axis: Int = 0) -> Tensor[T]:
    var t_new = Tensor[T](x.shape()[1 - axis])

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        if axis == 0:
            t_new[j] += x[i][j]
        else:
            t_new[i] += x[i][j]

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return t_new
