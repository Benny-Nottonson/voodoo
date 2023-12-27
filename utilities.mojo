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

fn axis_sum[T: DType](x: Tensor[T], axis: Int) -> Tensor[T]:
    var t_new = Tensor[T](1, x.shape()[1])

    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        let i = idx // x.shape()[1]
        let j = idx % x.shape()[1]
        t_new[0][j] += x[i][j]

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return t_new
    