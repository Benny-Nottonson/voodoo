from voodoo import Tensor


fn mse(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mse(expected)


fn mae(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mae(expected)


fn mape(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mape(expected)


fn msle(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.msle(expected)


fn bce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.bce(expected)


fn cce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.cce(expected)


fn cfce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.cfce(expected)
