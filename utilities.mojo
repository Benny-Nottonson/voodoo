from algorithm import vectorize

fn transposition[T: DType](x: Tensor[T]) -> Tensor[T]:
    var t_new = Tensor[T](x.shape())

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        t_new[i][j] = x[j][i]

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return t_new

fn axis_sum[T: DType](x: Tensor[T], axis: Int = 0) -> Tensor[T]:
    var t_new = Tensor[T](x.shape()[axis])

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx
        var sum: SIMD[T, 1] = 0.0
        for j in range(x.shape()[1 - axis]):
            if axis == 0:
                sum += x[j][i]
            else:
                sum += x[i][j]
        t_new[i] = sum

    vectorize[simdwidthof[T](), vecmath](t_new.num_elements())
    return t_new
    