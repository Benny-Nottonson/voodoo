from voodoo import (
    Tensor
)

"""
Loss Functions
- kubler_loss_divergence
- mean_absolute_error
- mean_absolute_percentage_error
- mean_squared_error
- mean_squared_logarithmic_error
- binary_cross_entropy
- categorical_cross_entropy
- categorical_focal_cross_entropy
- huber_loss
- log_cosh_error
- poisson_loss
- sparse_categorical_cross_entropy
"""

fn kld(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.kld(expected)


fn mae(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mae(expected)


fn mape(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mape(expected)


fn mse(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.mse(expected)


fn msle(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.msle(expected)


fn bce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.bce(expected)


fn cce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.cce(expected)


fn cfce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.cfce(expected)


fn cs(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.cs(expected)


fn huber(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.huber(expected)


fn logcosh(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.logcosh(expected)


fn poisson(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.poisson(expected)


fn scce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.scce(expected)