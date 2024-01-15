from .operations import (
    Copy,
    Reshape,
    Transpose,
    Sum,
    Dropout,
)
from .binary_operations import MMul
from .arithmetic import (
    Sqrt,
    Abs,
    Exp2,
    Log2,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
)
from .binary_arithmetic import (
    Add,
    Mul,
    Sub,
    Div,
    Pow,
)

from .regression_losses import (
    MSE,
    MAE,
    MAPE,
    MSLE,
)

from .activations import (
    Relu,
    Sigmoid,
    Softplus,
    Softsign,
    Tanh,
    Selu,
    Elu,
    Exp,
    LeakyRelu,
    Relu6,
    Silu,
    Gelu,
    HardSigmoid,
    Linear,
    Mish,
)


fn load_kernels() -> Pointer[op_tuple]:
    let kernels = Pointer[op_tuple].alloc(100)
    kernels.store(copy_code, op_tuple(Copy.fw, _b))
    kernels.store(copy_code + 1, op_tuple(Copy.bw, _b))
    kernels.store(reshape_code, op_tuple(Reshape.fw, _b))
    kernels.store(reshape_code + 1, op_tuple(Reshape.bw, _b))
    kernels.store(transp_code, op_tuple(Transpose.fw, _b))
    kernels.store(transp_code + 1, op_tuple(Transpose.bw, _b))
    kernels.store(sum_code, op_tuple(Sum.fw, _b))
    kernels.store(sum_code + 1, op_tuple(Sum.bw, _b))
    kernels.store(dropout_code, op_tuple(Dropout.fw, _b))
    kernels.store(dropout_code + 1, op_tuple(Dropout.bw, _b))
    kernels.store(mmul_code, op_tuple(_u, MMul.fw))
    kernels.store(mmul_code + 1, op_tuple(_u, MMul.bw))
    kernels.store(sqrt_code, op_tuple(Sqrt.fw, _b))
    kernels.store(sqrt_code + 1, op_tuple(Sqrt.bw, _b))
    kernels.store(abs_code, op_tuple(Abs.fw, _b))
    kernels.store(abs_code + 1, op_tuple(Abs.bw, _b))
    kernels.store(exp2_code, op_tuple(Exp2.fw, _b))
    kernels.store(exp2_code + 1, op_tuple(Exp2.bw, _b))
    kernels.store(log2_code, op_tuple(Log2.fw, _b))
    kernels.store(log2_code + 1, op_tuple(Log2.bw, _b))
    kernels.store(log_code, op_tuple(Log.fw, _b))
    kernels.store(log_code + 1, op_tuple(Log.bw, _b))
    kernels.store(sin_code, op_tuple(Sin.fw, _b))
    kernels.store(sin_code + 1, op_tuple(Sin.bw, _b))
    kernels.store(cos_code, op_tuple(Cos.fw, _b))
    kernels.store(cos_code + 1, op_tuple(Cos.bw, _b))
    kernels.store(tan_code, op_tuple(Tan.fw, _b))
    kernels.store(tan_code + 1, op_tuple(Tan.bw, _b))
    kernels.store(asin_code, op_tuple(Asin.fw, _b))
    kernels.store(asin_code + 1, op_tuple(Asin.bw, _b))
    kernels.store(acos_code, op_tuple(Acos.fw, _b))
    kernels.store(acos_code + 1, op_tuple(Acos.bw, _b))
    kernels.store(atan_code, op_tuple(Atan.fw, _b))
    kernels.store(atan_code + 1, op_tuple(Atan.bw, _b))
    kernels.store(sinh_code, op_tuple(Sinh.fw, _b))
    kernels.store(sinh_code + 1, op_tuple(Sinh.bw, _b))
    kernels.store(cosh_code, op_tuple(Cosh.fw, _b))
    kernels.store(cosh_code + 1, op_tuple(Cosh.bw, _b))
    kernels.store(add_code, op_tuple(_u, Add.fw))
    kernels.store(add_code + 1, op_tuple(_u, Add.bw))
    kernels.store(mul_code, op_tuple(_u, Mul.fw))
    kernels.store(mul_code + 1, op_tuple(_u, Mul.bw))
    kernels.store(sub_code, op_tuple(_u, Sub.fw))
    kernels.store(sub_code + 1, op_tuple(_u, Sub.bw))
    kernels.store(div_code, op_tuple(_u, Div.fw))
    kernels.store(div_code + 1, op_tuple(_u, Div.bw))
    kernels.store(pow_code, op_tuple(_u, Pow.fw))
    kernels.store(pow_code + 1, op_tuple(_u, Pow.bw))
    kernels.store(mse_code, op_tuple(_u, MSE.fw))
    kernels.store(mse_code + 1, op_tuple(_u, MSE.bw))
    kernels.store(mae_code, op_tuple(_u, MAE.fw))
    kernels.store(mae_code + 1, op_tuple(_u, MAE.bw))
    kernels.store(mape_code, op_tuple(_u, MAPE.fw))
    kernels.store(mape_code + 1, op_tuple(_u, MAPE.bw))
    kernels.store(msle_code, op_tuple(_u, MSLE.fw))
    kernels.store(msle_code + 1, op_tuple(_u, MSLE.bw))
    kernels.store(relu_code, op_tuple(Relu[0.0, f32_max, 0.0].fw, _b))
    kernels.store(relu_code + 1, op_tuple(Relu[0.0, f32_max, 0.0].bw, _b))
    kernels.store(sigmoid_code, op_tuple(Sigmoid.fw, _b))
    kernels.store(sigmoid_code + 1, op_tuple(Sigmoid.bw, _b))
    kernels.store(softplus_code, op_tuple(Softplus.fw, _b))
    kernels.store(softplus_code + 1, op_tuple(Softplus.bw, _b))
    kernels.store(softsign_code, op_tuple(Softsign.fw, _b))
    kernels.store(softsign_code + 1, op_tuple(Softsign.bw, _b))
    kernels.store(tanh_code, op_tuple(Tanh.fw, _b))
    kernels.store(tanh_code + 1, op_tuple(Tanh.bw, _b))
    kernels.store(selu_code, op_tuple(Selu.fw, _b))
    kernels.store(selu_code + 1, op_tuple(Selu.bw, _b))
    kernels.store(elu_code, op_tuple(Elu[0.0].fw, _b))
    kernels.store(elu_code + 1, op_tuple(Elu[0.0].bw, _b))
    kernels.store(exp_code, op_tuple(Exp.fw, _b))
    kernels.store(exp_code + 1, op_tuple(Exp.bw, _b))
    kernels.store(lrelu_code, op_tuple(LeakyRelu[0.0].fw, _b))
    kernels.store(lrelu_code + 1, op_tuple(LeakyRelu[0.0].bw, _b))
    kernels.store(relu6_code, op_tuple(Relu6.fw, _b))
    kernels.store(relu6_code + 1, op_tuple(Relu6.bw, _b))
    kernels.store(silu_code, op_tuple(Silu.fw, _b))
    kernels.store(silu_code + 1, op_tuple(Silu.bw, _b))
    kernels.store(gelu_code, op_tuple(Gelu[0.0].fw, _b))
    kernels.store(gelu_code + 1, op_tuple(Gelu[0.0].bw, _b))
    kernels.store(h_sig_code, op_tuple(HardSigmoid.fw, _b))
    kernels.store(h_sig_code + 1, op_tuple(HardSigmoid.bw, _b))
    kernels.store(linear_code, op_tuple(Linear.fw, _b))
    kernels.store(linear_code + 1, op_tuple(Linear.bw, _b))
    kernels.store(mish_code, op_tuple(Mish.fw, _b))
    kernels.store(mish_code + 1, op_tuple(Mish.bw, _b))
    return kernels
