from tensor import TensorShape
from initializer import Initializer
from regularizer import Regularizer
from activation import Activation
from loss import Loss, LossFunction
from utilities import transposition, axis_sum


@value
struct Conv1D[
    T: DType,
    inputChannels: Int,
    outputChannels: Int,
    kernelLength: Int,
    strides: Int = 1,
    initializer_type: String = "random_uniform",
    regularizer_type: String = "random_uniform", # TODO
    activation_type: String = "relu",
    loss_function_type: String = "mse",
]:
    var activation: Activation[T, activation_type]
    var lossFunction: LossFunction[T, loss_function_type]
    var weights: Tensor[T]
    var bias: Tensor[T]

    var input: Tensor[T]
    var z: Tensor[T]
    var dw: Tensor[T]
    var db: Tensor[T]

    fn __init__(inout self) raises:
        self.activation = Activation[T, activation_type]()
        self.lossFunction = LossFunction[T, loss_function_type]()
        
        self.weights = Tensor[T](inputChannels, outputChannels, kernelLength)
        Initializer[T, initializer_type]().initialize(self.weights)
        self.bias = Tensor[T](outputChannels)
        Initializer[T, "zeros"]().initialize(self.bias)

        self.input = Tensor[T]()
        self.z = Tensor[T]()
        self.dw = Tensor[T]()
        self.db = Tensor[T]()

    fn forward(inout self, input: Tensor[T]) raises -> Tensor[T]:
        self.input = input
        self.z = conv1d[T](input, self.weights, self.bias, self.strides)
        return self.activation.forward(self.z)

    # TODO: backward


fn conv1d[T: DType](
    input: Tensor[T],
    weights: Tensor[T],
    bias: Tensor[T],
    strides: Int = 1,
) raises -> Tensor[T]:
    let inputShape = input.shape()
    let batchSize = inputShape[0]
    let inputLength = inputShape[1]
    let inputChannels = inputShape[2]

    let weightsShape = weights.shape()
    let outputChannels = weightsShape[1]
    let kernelLength = weightsShape[2]

    let outputLength = ((inputLength - kernelLength) / strides + 1).to_int()

    var output = Tensor[T](batchSize, outputLength, outputChannels)

    for b in range(batchSize):
        for c in range(outputChannels):
            for i in range(outputLength):
                for j in range(kernelLength):
                    output[b][i][c] += input[b][i * strides + j][c] * weights[0][c][j]
                output[b][i][c] += bias[0][c]
    return output