from .operations import (
    Copy,
    Reshape,
    Transpose,
    Sum,
    Dropout,
)
from .matmul import MMul
from .maxpool import MaxPool1D, MaxPool2D
from .conv import Conv1D, Conv2D
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
    Add,
    Mul,
    Sub,
    Div,
    Pow,
)
from .losses import MSE, MAE, MAPE, MSLE
from ..operator_codes import *
from ..constants import F32_MAX, UNARY_OP, BINARY_OP, OP_TUPLE, NU, NB


fn k_add[
    code: Int,
    u_fw: UNARY_OP,
    u_bw: UNARY_OP,
](kernel: Pointer[OP_TUPLE]):
    kernel.store(code, OP_TUPLE(u_fw, NB))
    kernel.store(code + 1, OP_TUPLE(u_bw, NB))


fn k_add[
    code: Int,
    b_fw: BINARY_OP,
    bN_Bw: BINARY_OP,
](kernel: Pointer[OP_TUPLE]):
    kernel.store(code, OP_TUPLE(NU, b_fw))
    kernel.store(code + 1, OP_TUPLE(NU, bN_Bw))


fn load_kernels() -> Pointer[OP_TUPLE]:
    let kernels = Pointer[OP_TUPLE].alloc(100)
    k_add[copy_code, Copy.fw, Copy.bw](kernels)
    k_add[reshape_code, Reshape.fw, Reshape.bw](kernels)
    k_add[transp_code, Transpose.fw, Transpose.bw](kernels)
    k_add[sum_code, Sum.fw, Sum.bw](kernels)
    k_add[dropout_code, Dropout.fw, Dropout.bw](kernels)
    k_add[mmul_code, MMul.fw, MMul.bw](kernels)
    k_add[sqrt_code, Sqrt.fw, Sqrt.bw](kernels)
    k_add[abs_code, Abs.fw, Abs.bw](kernels)
    k_add[exp2_code, Exp2.fw, Exp2.bw](kernels)
    k_add[log2_code, Log2.fw, Log2.bw](kernels)
    k_add[log_code, Log.fw, Log.bw](kernels)
    k_add[sin_code, Sin.fw, Sin.bw](kernels)
    k_add[cos_code, Cos.fw, Cos.bw](kernels)
    k_add[tan_code, Tan.fw, Tan.bw](kernels)
    k_add[asin_code, Asin.fw, Asin.bw](kernels)
    k_add[acos_code, Acos.fw, Acos.bw](kernels)
    k_add[atan_code, Atan.fw, Atan.bw](kernels)
    k_add[sinh_code, Sinh.fw, Sinh.bw](kernels)
    k_add[cosh_code, Cosh.fw, Cosh.bw](kernels)
    k_add[add_code, Add.fw, Add.bw](kernels)
    k_add[mul_code, Mul.fw, Mul.bw](kernels)
    k_add[sub_code, Sub.fw, Sub.bw](kernels)
    k_add[div_code, Div.fw, Div.bw](kernels)
    k_add[pow_code, Pow.fw, Pow.bw](kernels)
    k_add[mse_code, MSE.fw, MSE.bw](kernels)
    k_add[mae_code, MAE.fw, MAE.bw](kernels)
    k_add[mape_code, MAPE.fw, MAPE.bw](kernels)
    k_add[msle_code, MSLE.fw, MSLE.bw](kernels)
    k_add[relu_code, Relu[0.0, F32_MAX, 0.0].fw, Relu[0.0, F32_MAX, 0.0].bw](kernels)
    k_add[sigmoid_code, Sigmoid.fw, Sigmoid.bw](kernels)
    k_add[softplus_code, Softplus.fw, Softplus.bw](kernels)
    k_add[softsign_code, Softsign.fw, Softsign.bw](kernels)
    k_add[tanh_code, Tanh.fw, Tanh.bw](kernels)
    k_add[selu_code, Selu.fw, Selu.bw](kernels)
    k_add[elu_code, Elu[0.0].fw, Elu[0.0].bw](kernels)
    k_add[exp_code, Exp.fw, Exp.bw](kernels)
    k_add[lrelu_code, LeakyRelu[0.0].fw, LeakyRelu[0.0].bw](kernels)
    k_add[relu6_code, Relu6.fw, Relu6.bw](kernels)
    k_add[silu_code, Silu.fw, Silu.bw](kernels)
    k_add[gelu_code, Gelu[0.0].fw, Gelu[0.0].bw](kernels)
    k_add[h_sig_code, HardSigmoid.fw, HardSigmoid.bw](kernels)
    k_add[linear_code, Linear.fw, Linear.bw](kernels)
    k_add[mish_code, Mish.fw, Mish.bw](kernels)
    k_add[conv1d_code, Conv1D.fw, Conv1D.bw](kernels)
    k_add[conv2d_code, Conv2D.fw, Conv2D.bw](kernels)
    k_add[maxpool1d_code, MaxPool1D.fw, MaxPool1D.bw](kernels)
    k_add[maxpool2d_code, MaxPool2D.fw, MaxPool2D.bw](kernels)
    return kernels
