fn get_activation_code[name: String]() -> Int:
    @parameter
    if name == "relu":
        return relu_code
    elif name == "sigmoid":
        return sigmoid_code
    elif name == "softplus":
        return softplus_code
    elif name == "softsign":
        return softsign_code
    elif name == "tanh":
        return tanh_code
    elif name == "selu":
        return selu_code
    elif name == "elu":
        return elu_code
    elif name == "exp":
        return exp_code
    elif name == "lrelu":
        return lrelu_code
    elif name == "relu6":
        return relu6_code
    elif name == "silu":
        return silu_code
    elif name == "gelu":
        return gelu_code
    elif name == "h_sig":
        return h_sig_code
    elif name == "linear":
        return linear_code
    elif name == "mish":
        return mish_code
    warn("Invalid activation function: " + name + " using linear\n")
    return linear_code


fn get_loss_code[name: String]() -> Int:
    @parameter
    if name == "mse":
        return mse_code
    elif name == "mae":
        return mae_code
    elif name == "mape":
        return mape_code
    elif name == "msle":
        return msle_code
    warn("Invalid loss function: " + name + " using mse\n")
    return mse_code
