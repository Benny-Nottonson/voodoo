from voodoo import Tensor

"""
Initializer Functions
- glorot_normal
- glorot_uniform
- he_normal
- he_uniform
- identity
- lecun_normal
- lecun_uniform
- ones
- random_normal
- random_uniform
- truncated_normal
- zeros
- constant
"""


fn glorot_normal(tensor: Tensor) raises -> Tensor:
    return tensor.glorot_normal()


fn glorot_uniform(tensor: Tensor) raises -> Tensor:
    return tensor.glorot_uniform()


fn he_normal(tensor: Tensor) raises -> Tensor:
    return tensor.he_normal()


fn he_uniform(tensor: Tensor) raises -> Tensor:
    return tensor.he_uniform()


fn identity(tensor: Tensor) raises -> Tensor:
    return tensor.identity()


fn lecun_normal(tensor: Tensor) raises -> Tensor:
    return tensor.lecun_normal()


fn lecun_uniform(tensor: Tensor) raises -> Tensor:
    return tensor.lecun_uniform()


fn ones(tensor: Tensor) raises -> Tensor:
    return tensor.ones()


fn random_normal(tensor: Tensor) raises -> Tensor:
    return tensor.random_normal()


fn random_uniform(tensor: Tensor, min: Float32, max: Float32) raises -> Tensor:
    return tensor.random_uniform(min, max)


fn truncated_normal(tensor: Tensor) raises -> Tensor:
    return tensor.truncated_normal()


fn zeros(tensor: Tensor) raises -> Tensor:
    return tensor.zeros()


fn constant(tensor: Tensor, value: Float32) raises -> Tensor:
    return tensor.fill(value)


fn _custom_fill(tensor: Tensor, values: DynamicVector[Float32]) raises -> Tensor:
    return tensor._custom_fill(values)
