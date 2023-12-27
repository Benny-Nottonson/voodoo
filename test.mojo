from layer import ActivationLayer
from tensor import TensorShape
from random import rand

fn main() raises:
    let testTensor = rand[DType.float64](TensorShape(2, 2))

    let testLayer = ActivationLayer[DType.float64, "relu", TensorShape(2, 2)]()

    print(testLayer.forward(testTensor))