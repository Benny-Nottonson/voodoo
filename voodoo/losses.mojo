from voodoo import Tensor

# TODO: Eventually can be a dict, not in mojo yet
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