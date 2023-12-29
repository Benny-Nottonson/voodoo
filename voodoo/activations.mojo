from voodoo import Tensor


fn elu(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[elu_code]()


fn exp(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[exp_code]()


fn gelu(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[gelu_code]()


fn h_sig(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[h_sig_code]()


fn linear(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[linear_code]()


fn mish(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[mish_code]()


fn relu(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[relu_code]()


fn selu(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[selu_code]()


fn sig(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[sig_code]()


fn softmax(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[softmax_code]()


fn softplus(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[softplus_code]()


fn softsign(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[softsign_code]()


fn swish(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[swish_code]()


fn tanh(tensor: Tensor) raises -> Tensor:
    return tensor.compute_activation[tanh_code]()
