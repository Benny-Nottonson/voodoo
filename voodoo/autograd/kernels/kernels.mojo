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

from voodoo.constants import F32_MAX, UNARY_OP, BINARY_OP, OP_TUPLE, NU, NB
from voodoo.utils.operator_codes import *


@register_passable("trivial")
struct KERNELS:
    @staticmethod
    fn get(code: Int) -> OP_TUPLE:
        if code == copy_code:
            return OP_TUPLE(Copy.fw, NB)
        elif code == copy_code + 1:
            return OP_TUPLE(Copy.bw, NB)
        elif code == reshape_code:
            return OP_TUPLE(Reshape.fw, NB)
        elif code == reshape_code + 1:
            return OP_TUPLE(Reshape.bw, NB)
        elif code == transp_code:
            return OP_TUPLE(Transpose.fw, NB)
        elif code == transp_code + 1:
            return OP_TUPLE(Transpose.bw, NB)
        elif code == sum_code:
            return OP_TUPLE(Sum.fw, NB)
        elif code == sum_code + 1:
            return OP_TUPLE(Sum.bw, NB)
        elif code == dropout_code:
            return OP_TUPLE(Dropout.fw, NB)
        elif code == dropout_code + 1:
            return OP_TUPLE(Dropout.bw, NB)
        elif code == mmul_code:
            return OP_TUPLE(NU, MMul.fw)
        elif code == mmul_code + 1:
            return OP_TUPLE(NU, MMul.bw)
        elif code == sqrt_code:
            return OP_TUPLE(Sqrt.fw, NB)
        elif code == sqrt_code + 1:
            return OP_TUPLE(Sqrt.bw, NB)
        elif code == abs_code:
            return OP_TUPLE(Abs.fw, NB)
        elif code == abs_code + 1:
            return OP_TUPLE(Abs.bw, NB)
        elif code == exp2_code:
            return OP_TUPLE(Exp2.fw, NB)
        elif code == exp2_code + 1:
            return OP_TUPLE(Exp2.bw, NB)
        elif code == log2_code:
            return OP_TUPLE(Log2.fw, NB)
        elif code == log2_code + 1:
            return OP_TUPLE(Log2.bw, NB)
        elif code == log_code:
            return OP_TUPLE(Log.fw, NB)
        elif code == log_code + 1:
            return OP_TUPLE(Log.bw, NB)
        elif code == sin_code:
            return OP_TUPLE(Sin.fw, NB)
        elif code == sin_code + 1:
            return OP_TUPLE(Sin.bw, NB)
        elif code == cos_code:
            return OP_TUPLE(Cos.fw, NB)
        elif code == cos_code + 1:
            return OP_TUPLE(Cos.bw, NB)
        elif code == tan_code:
            return OP_TUPLE(Tan.fw, NB)
        elif code == tan_code + 1:
            return OP_TUPLE(Tan.bw, NB)
        elif code == asin_code:
            return OP_TUPLE(Asin.fw, NB)
        elif code == asin_code + 1:
            return OP_TUPLE(Asin.bw, NB)
        elif code == acos_code:
            return OP_TUPLE(Acos.fw, NB)
        elif code == acos_code + 1:
            return OP_TUPLE(Acos.bw, NB)
        elif code == atan_code:
            return OP_TUPLE(Atan.fw, NB)
        elif code == atan_code + 1:
            return OP_TUPLE(Atan.bw, NB)
        elif code == sinh_code:
            return OP_TUPLE(Sinh.fw, NB)
        elif code == sinh_code + 1:
            return OP_TUPLE(Sinh.bw, NB)
        elif code == cosh_code:
            return OP_TUPLE(Cosh.fw, NB)
        elif code == cosh_code + 1:
            return OP_TUPLE(Cosh.bw, NB)
        elif code == add_code:
            return OP_TUPLE(NU, Add.fw)
        elif code == add_code + 1:
            return OP_TUPLE(NU, Add.bw)
        elif code == mul_code:
            return OP_TUPLE(NU, Mul.fw)
        elif code == mul_code + 1:
            return OP_TUPLE(NU, Mul.bw)
        elif code == sub_code:
            return OP_TUPLE(NU, Sub.fw)
        elif code == sub_code + 1:
            return OP_TUPLE(NU, Sub.bw)
        elif code == div_code:
            return OP_TUPLE(NU, Div.fw)
        elif code == div_code + 1:
            return OP_TUPLE(NU, Div.bw)
        elif code == pow_code:
            return OP_TUPLE(NU, Pow.fw)
        elif code == pow_code + 1:
            return OP_TUPLE(NU, Pow.bw)
        elif code == mse_code:
            return OP_TUPLE(NU, MSE.fw)
        elif code == mse_code + 1:
            return OP_TUPLE(NU, MSE.bw)
        elif code == mae_code:
            return OP_TUPLE(NU, MAE.fw)
        elif code == mae_code + 1:
            return OP_TUPLE(NU, MAE.bw)
        elif code == mape_code:
            return OP_TUPLE(NU, MAPE.fw)
        elif code == mape_code + 1:
            return OP_TUPLE(NU, MAPE.bw)
        elif code == msle_code:
            return OP_TUPLE(NU, MSLE.fw)
        elif code == msle_code + 1:
            return OP_TUPLE(NU, MSLE.bw)
        elif code == relu_code:
            return OP_TUPLE(Relu[0.0, F32_MAX, 0.0].fw, NB)
        elif code == relu_code + 1:
            return OP_TUPLE(Relu[0.0, F32_MAX, 0.0].bw, NB)
        elif code == sigmoid_code:
            return OP_TUPLE(Sigmoid.fw, NB)
        elif code == sigmoid_code + 1:
            return OP_TUPLE(Sigmoid.bw, NB)
        elif code == softplus_code:
            return OP_TUPLE(Softplus.fw, NB)
        elif code == softplus_code + 1:
            return OP_TUPLE(Softplus.bw, NB)
        elif code == softsign_code:
            return OP_TUPLE(Softsign.fw, NB)
        elif code == softsign_code + 1:
            return OP_TUPLE(Softsign.bw, NB)
        elif code == tanh_code:
            return OP_TUPLE(Tanh.fw, NB)
        elif code == tanh_code + 1:
            return OP_TUPLE(Tanh.bw, NB)
        elif code == selu_code:
            return OP_TUPLE(Selu.fw, NB)
        elif code == selu_code + 1:
            return OP_TUPLE(Selu.bw, NB)
        elif code == elu_code:
            return OP_TUPLE(Elu[0.0].fw, NB)
        elif code == elu_code + 1:
            return OP_TUPLE(Elu[0.0].bw, NB)
        elif code == exp_code:
            return OP_TUPLE(Exp.fw, NB)
        elif code == exp_code + 1:
            return OP_TUPLE(Exp.bw, NB)
        elif code == lrelu_code:
            return OP_TUPLE(LeakyRelu[0.0].fw, NB)
        elif code == lrelu_code + 1:
            return OP_TUPLE(LeakyRelu[0.0].bw, NB)
        elif code == relu6_code:
            return OP_TUPLE(Relu6.fw, NB)
        elif code == relu6_code + 1:
            return OP_TUPLE(Relu6.bw, NB)
        elif code == silu_code:
            return OP_TUPLE(Silu.fw, NB)
        elif code == silu_code + 1:
            return OP_TUPLE(Silu.bw, NB)
        elif code == gelu_code:
            return OP_TUPLE(Gelu[0.0].fw, NB)
        elif code == gelu_code + 1:
            return OP_TUPLE(Gelu[0.0].bw, NB)
        elif code == h_sig_code:
            return OP_TUPLE(HardSigmoid.fw, NB)
        elif code == h_sig_code + 1:
            return OP_TUPLE(HardSigmoid.bw, NB)
        elif code == linear_code:
            return OP_TUPLE(Linear.fw, NB)
        elif code == linear_code + 1:
            return OP_TUPLE(Linear.bw, NB)
        elif code == mish_code:
            return OP_TUPLE(Mish.fw, NB)
        elif code == mish_code + 1:
            return OP_TUPLE(Mish.bw, NB)
        elif code == conv1d_code:
            return OP_TUPLE(NU, Conv1D.fw)
        elif code == conv1d_code + 1:
            return OP_TUPLE(NU, Conv1D.bw)
        elif code == conv2d_code:
            return OP_TUPLE(NU, Conv2D.fw)
        elif code == conv2d_code + 1:
            return OP_TUPLE(NU, Conv2D.bw)
        elif code == maxpool1d_code:
            return OP_TUPLE(MaxPool1D.fw, NB)
        elif code == maxpool1d_code + 1:
            return OP_TUPLE(MaxPool1D.bw, NB)
        elif code == maxpool2d_code:
            return OP_TUPLE(MaxPool2D.fw, NB)
        elif code == maxpool2d_code + 1:
            return OP_TUPLE(MaxPool2D.bw, NB)
        else:
            return OP_TUPLE(NU, NB)
