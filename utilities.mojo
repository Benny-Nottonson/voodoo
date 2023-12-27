from algorithm import vectorize


fn map[T: DType](x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> Tensor[T]:
    var new_tensor = Tensor[T](x.shape())
    
    @parameter
    fn vecmath[simd_width: Int](idx: Int) -> None:
        new_tensor[idx] = f(x[idx])

    vectorize[simdwidthof[T](), vecmath](x.num_elements())
    return new_tensor


fn apply[T: DType](owned x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> None:
    for i in range(x.num_elements()):
        x[i] = f(x[i])


fn reduce[T: DType](x: Tensor[T], f: fn (SIMD[T, 1]) -> SIMD[T, 1]) -> SIMD[T, 1]:
    var result = f(x[0])
    for i in range(1, x.num_elements()):
        result += f(x[i])
    return result
    


fn reduce[
    T: DType
](
    f: fn (SIMD[T, 1], SIMD[T, 1]) -> SIMD[T, 1], x: Tensor[T], y: Tensor[T], axis: Int
) raises -> Tensor[T]:
    var new_tensor = Tensor[T](x.shape())
    for i in range(x.shape()[axis]):
        new_tensor[i] = f(x[i], y[i])
    var newShape = DynamicVector[Int]()
    for i in range(x.rank()):
        if i != axis:
            newShape.append(x.shape()[i])
    return new_tensor.reshape(newShape)
    


fn transposition[T: DType](x: Tensor[T]) -> Tensor[T]:
    var new_tensor = Tensor[T](x.shape())
    for i in range(x.shape()[0]):
        for j in range(x.shape()[1]):
            new_tensor[j][i] = x[i][j]
    return new_tensor
   


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

