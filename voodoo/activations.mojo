from voodoo import Tensor

# https://github.com/Benny-Nottonson/voodoo/wiki/Activation-Functions


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


fn softmax(tensor: Tensor, axis: Int) raises -> Tensor:
    return tensor.softmax(axis)


fn softmax(tensor: Tensor) raises -> Tensor:
    return tensor.softmax(-1)


fn softplus(tensor: Tensor) raises -> Tensor:
    return tensor.softplus()


fn softsign(tensor: Tensor) raises -> Tensor:
    return tensor.softsign()


fn swish(tensor: Tensor) raises -> Tensor:
    return tensor.swish()


fn tanh(tensor: Tensor) raises -> Tensor:
    return tensor.tanh()
