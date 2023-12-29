from voodoo import Tensor


fn mae(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=mae_code](expected)


fn mape(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=mape_code](expected)


fn mse(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=mse_code](expected)


fn msle(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=msle_code](expected)


fn bce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=bce_code](expected)


fn cce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=cce_code](expected)


fn cfce(predicted: Tensor, expected: Tensor) raises -> Tensor:
    return predicted.compute_loss[operator_id=cfce_code](expected)
