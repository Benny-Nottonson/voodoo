from voodoo import Tensor

"""
Activation Functions
- elu
- exp
- gelu
- hard_sigmoid
- linear
- mish
- relu
- selu
- sigmoid
- softmax
- softplus
- softsign
- swish
- tanh
"""


fn elu(tensor: Tensor) raises -> Tensor:
    return tensor.elu()


fn exp(tensor: Tensor) raises -> Tensor:
    return tensor.exp()


fn gelu(tensor: Tensor) raises -> Tensor:
    return tensor.gelu()


fn hard_sigmoid(tensor: Tensor) raises -> Tensor:
    return tensor.hard_sigmoid()


fn linear(tensor: Tensor) raises -> Tensor:
    return tensor.linear()


fn mish(tensor: Tensor) raises -> Tensor:
    return tensor.mish()


fn relu(tensor: Tensor) raises -> Tensor:
    return tensor.relu()


fn selu(tensor: Tensor) raises -> Tensor:
    return tensor.selu()


fn sigmoid(tensor: Tensor) raises -> Tensor:
    return tensor.sigmoid()


fn softmax(tensor: Tensor, axis: Int = -1) raises -> Tensor:
    return tensor.softmax(axis)


fn softplus(tensor: Tensor) raises -> Tensor:
    return tensor.softplus()


fn softsign(tensor: Tensor) raises -> Tensor:
    return tensor.softsign()


fn swish(tensor: Tensor) raises -> Tensor:
    return tensor.swish()


fn tanh(tensor: Tensor) raises -> Tensor:
    return tensor.tanh()
