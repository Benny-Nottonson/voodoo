struct Regularizer:
    var name: String

    fn __init__(inout self, name: String):
        self.name = name

    fn apply[T: DType](self, x: Tensor[T]) -> Tensor[T]:
        if self.name == "dropout":
            return Dropout.apply(x)
        """
        elif self.name == "l1":
            return L1.apply(x)
        elif self.name == "l2":
            return L2.apply(x)
        """
        return x

@value
struct Dropout:
    @staticmethod 
    fn apply[T: DType](x: Tensor[T]) -> Tensor[T]:
        return x #TODO