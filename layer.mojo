from tensor import TensorShape
from initializer import Initializer
from regularizer import Regularizer
from activation import Activation
from loss import Loss
from utilities import transposition, axis_sum


@value
struct Layer[
    T: DType,
    name: String,
    inputShape: TensorShape,
    outputShape: TensorShape,
    initializer_type: String,
    regularizer_type: String,
    activation_type: String,
]:
    var activation: Activation[T, activation_type]

    var weights: Tensor[T]
    var bias: Tensor[T]

    var input: Tensor[T]
    var z: Tensor[T]
    var dw: Tensor[T]
    var db: Tensor[T]

    var forward: fn (x: Tensor[T]) -> Tensor[T]
    var backward: fn (x: Tensor[T], y: Tensor[T]) -> Tensor[T]

    fn __init__(inout self) raises:
        self.activation = Activation[T, activation_type]()

        var weightShape = DynamicVector[Int]()
        for i in range(inputShape.rank()):
            weightShape.append(inputShape[i])
        for i in range(outputShape.rank()):
            weightShape.append(outputShape[i])

        self.weights = Tensor[T](TensorShape(weightShape))
        self.bias = Tensor[T](outputShape)

        self.input = Tensor[T](inputShape)
        self.z = Tensor[T](outputShape)
        self.dw = Tensor[T](TensorShape(weightShape))
        self.db = Tensor[T](outputShape)

        if self.name == "conv1d":
            self.forward = Conv1D.forward[T]
            self.backward = Conv1D.backward[T]
        elif self.name == "dense":
            self.forward = self.dense_forward
            self.backward = self.dense_backward
        else:
            raise Error("Invalid layer name")


@value
struct Conv1D:
    @staticmethod
    fn forward[T: DType](x: Tensor[T], w: Tensor[T], b: Tensor[T]) raises -> Tensor[T]:
        var (m, n_H_prev, n_W_prev, n_C_prev) = x.shape
        var (f, n_C_prev, n_C) = w.shape

        var n_H = n_H_prev
        var n_W = n_W_prev

        var z = Tensor[T](TensorShape(m, n_H, n_W, n_C))

        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        var vert_start = h
                        var vert_end = h + f
                        var horiz_start = w
                        var horiz_end = w + f

                        var a_slice_prev = x[
                            i, vert_start:vert_end, horiz_start:horiz_end, :
                        ]
                        z[i, h, w, c] = (
                            axis_sum(a_slice_prev * w[:, :, :, c]) + b[:, :, :, c]
                        )

        return z
