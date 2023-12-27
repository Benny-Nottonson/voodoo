from tensor import TensorShape

@value
struct Input[
    DType: DType,
    shape: TensorShape,
    tensor: Tensor[DType] = Tensor[DType](shape),
    batch_shape: Int = 1,
    name: String = "",
]:
    ...
