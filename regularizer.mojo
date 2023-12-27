trait Regularizer:
    ...

@value
struct Empty(Regularizer):
    @staticmethod
    fn apply[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x

@value
struct Dropout(Regularizer):
    ...