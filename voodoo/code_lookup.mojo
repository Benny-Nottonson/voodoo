from voodoo import Tensor

fn get_activation_code[name: String]() -> Int:
    @parameter
    if name == "elu":
        return elu_code
    elif name == "exp":
        return exp_code
    elif name == "gelu":
        return gelu_code
    elif name == "h_sig":
        return h_sig_code
    elif name == "linear":
        return linear_code
    elif name == "mish":
        return mish_code
    elif name == "relu":
        return relu_code
    elif name == "selu":
        return selu_code
    elif name == "sig":
        return sig_code
    elif name == "softmax":
        return softmax_code
    elif name == "softplus":
        return softplus_code
    elif name == "softsign":
        return softsign_code
    elif name == "swish":
        return swish_code
    elif name == "tanh":
        return tanh_code
    elif name == "leaky_relu":
        return lrelu_code
    return linear_code

fn get_loss_code[name: String]() -> Int:
    @parameter
    if name == "mae":
        return mae_code
    elif name == "mape":
        return mape_code
    elif name == "mse":
        return mse_code
    elif name == "msle":
        return msle_code
    elif name == "bce":
        return bce_code
    elif name == "cce":
        return cce_code
    elif name == "cfce":
        return cfce_code
    return mse_code