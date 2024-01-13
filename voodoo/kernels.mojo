from .cpu_kernels.operations import (
    Copy,
    Reshape,
    Transpose,
    Sum,
    MaxPool2D,
    Dropout,
)
from .cpu_kernels.binary_operations import Conv2D, MMul
from .cpu_kernels.arithmetic import (
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
from .cpu_kernels.binary_arithmetic import (
    Add,
    Mul,
    Sub,
    Div,
    Pow,
)

from .cpu_kernels.regression_losses import (
    MSE,
    MAE,
    MAPE,
    MSLE,
)

from .cpu_kernels.activations import (
    Relu,
    Sigmoid,
    Softmax,
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
    LogSoftmax,
)

alias unary_op = fn (b: Node, a: Node) -> None
alias binary_op = fn (c: Node, a: Node, b: Node) -> None
alias op_tuple = Tuple[unary_op, binary_op]

# TODO: Update
# TODO: Convert all imports to explicit


fn _u(b: Node, a: Node):
    ...


fn _b(c: Node, a: Node, b: Node):
    ...


@register_passable("trivial")
struct KernelManager:
    var kernels: Pointer[op_tuple]

    fn __init__(kernels: Pointer[op_tuple]) -> Self:
        return KernelManager {kernels: kernels}

    fn store(self, opcode: Int, fw: unary_op, bw: unary_op):
        self.kernels.store(opcode, op_tuple(fw, _b))
        self.kernels.store(opcode + 1, op_tuple(bw, _b))

    fn store(self, opcode: Int, fw: binary_op, bw: binary_op):
        self.kernels.store(opcode, op_tuple(_u, fw))
        self.kernels.store(opcode + 1, op_tuple(_u, bw))


@register_passable("trivial")
struct Kernels:
    var kernels: Pointer[op_tuple]

    @always_inline
    fn __init__() -> Kernels:
        let kernels = Pointer[op_tuple].alloc(100)
        let k = KernelManager(kernels)
        k.store(copy_code, Copy.fw, Copy.bw)
        k.store(reshape_code, Reshape.fw, Reshape.bw)
        k.store(transp_code, Transpose.fw, Transpose.bw)
        k.store(sum_code, Sum.fw, Sum.bw)
        k.store(mpool2dd_code, MaxPool2D.fw, MaxPool2D.bw)
        k.store(dropout_code, Dropout.fw, Dropout.bw)
        k.store(conv2d_code, Conv2D.fw, Conv2D.bw)
        k.store(mmul_code, MMul.fw, MMul.bw)
        k.store(sqrt_code, Sqrt.fw, Sqrt.bw)
        k.store(abs_code, Abs.fw, Abs.bw)
        k.store(exp2_code, Exp2.fw, Exp2.bw)
        k.store(log2_code, Log2.fw, Log2.bw)
        k.store(log_code, Log.fw, Log.bw)
        k.store(sin_code, Sin.fw, Sin.bw)
        k.store(cos_code, Cos.fw, Cos.bw)
        k.store(tan_code, Tan.fw, Tan.bw)
        k.store(asin_code, Asin.fw, Asin.bw)
        k.store(acos_code, Acos.fw, Acos.bw)
        k.store(atan_code, Atan.fw, Atan.bw)
        k.store(sinh_code, Sinh.fw, Sinh.bw)
        k.store(cosh_code, Cosh.fw, Cosh.bw)
        k.store(add_code, Add.fw, Add.bw)
        k.store(mul_code, Mul.fw, Mul.bw)
        k.store(sub_code, Sub.fw, Sub.bw)
        k.store(div_code, Div.fw, Div.bw)
        k.store(pow_code, Pow.fw, Pow.bw)
        k.store(mse_code, MSE.fw, MSE.bw)
        k.store(mae_code, MAE.fw, MAE.bw)
        k.store(mape_code, MAPE.fw, MAPE.bw)
        k.store(msle_code, MSLE.fw, MSLE.bw)
        k.store(relu_code, Relu[0.0, f32_max, 0.0].fw, Relu[0.0, f32_max, 0.0].bw)
        k.store(sigmoid_code, Sigmoid.fw, Sigmoid.bw)
        k.store(softmax_code, Softmax.fw, Softmax.bw)
        k.store(softplus_code, Softplus.fw, Softplus.bw)
        k.store(softsign_code, Softsign.fw, Softsign.bw)
        k.store(tanh_code, Tanh.fw, Tanh.bw)
        k.store(selu_code, Selu.fw, Selu.bw)
        k.store(elu_code, Elu[0.0].fw, Elu[0.0].bw)
        k.store(exp_code, Exp.fw, Exp.bw)
        k.store(lrelu_code, LeakyRelu[0.0].fw, LeakyRelu[0.0].bw)
        k.store(relu6_code, Relu6.fw, Relu6.bw)
        k.store(silu_code, Silu.fw, Silu.bw)
        k.store(gelu_code, Gelu[0.0].fw, Gelu[0.0].bw)
        k.store(h_sig_code, HardSigmoid.fw, HardSigmoid.bw)
        k.store(linear_code, Linear.fw, Linear.bw)
        k.store(mish_code, Mish.fw, Mish.bw)
        k.store(log_softmax_code, LogSoftmax.fw, LogSoftmax.bw)
        return Kernels{kernels: kernels}
