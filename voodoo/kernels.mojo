from .cpu_kernels.operations import *
from .cpu_kernels.binary_operations import *
from .cpu_kernels.arithmetic import *
from .cpu_kernels.binary_arithmetic import *
from .cpu_kernels.activations import *
from .cpu_kernels.losses import *

alias unary_op = fn (b: Node, a: Node) -> None
alias binary_op = fn (c: Node, a: Node, b: Node) -> None
alias op_tuple = Tuple[String, unary_op, binary_op]


fn _u(b: Node, a: Node):
    ...


fn _b(c: Node, a: Node, b: Node):
    ...

@register_passable("trivial")
struct KernelManager:
    var kernels: Pointer[op_tuple]

    fn __init__(kernels: Pointer[op_tuple]) -> Self:
        return KernelManager {kernels: kernels}

    fn store(self, name: String, opcode: Int, fw: unary_op, bw: unary_op) -> Self:
        self.kernels.store(opcode, op_tuple(name, fw, _b))
        self.kernels.store(opcode + 1, op_tuple(name + "_bw", bw, _b))
        return self

    fn store(self, name: String, opcode: Int, fw: binary_op, bw: binary_op) -> Self:
        self.kernels.store(opcode, op_tuple(name, _u, fw))
        self.kernels.store(opcode + 1, op_tuple(name + "_bw", _u, bw))
        return self

@register_passable("trivial")
struct Kernels:
    var kernels: Pointer[op_tuple]

    fn __init__() -> Kernels:
        let kernels = Pointer[op_tuple].alloc(120)
        _ = KernelManager(kernels)
            .store("cos", cos_code, Cos.fw, Cos.bw)
            .store("sin", sin_code, Sin.fw, Sin.bw)
            .store("tan", tan_code, Tan.fw, Tan.bw)
            .store("acos", acos_code, Acos.fw, Acos.bw)
            .store("asin", asin_code, Asin.fw, Asin.bw)
            .store("atan", atan_code, Atan.fw, Atan.bw)
            .store("cosh", cosh_code, Cosh.fw, Cosh.bw)
            .store("sinh", sinh_code, Sinh.fw, Sinh.bw)
            .store("log", log_code, Log.fw, Log.bw)
            .store("log2", log2_code, Log2.fw, Log2.bw)
            .store("exp2", exp2_code, Exp2.fw, Exp2.bw)
            .store("sqrt", sqrt_code, Sqrt.fw, Sqrt.bw)
            .store("abs", abs_code, Abs.fw, Abs.bw)
            .store("copy", copy_code, Copy.fw, Copy.bw)
            .store("add", add_code, Add.fw, Add.bw)
            .store("sub", sub_code, Sub.fw, Sub.bw)
            .store("mul", mul_code, Mul.fw, Mul.bw)
            .store("div", div_code, Div.fw, Div.bw)
            .store("pow", pow_code, Pow.fw, Pow.bw)
            .store("mmul", mmul_code, MMul.fw, MMul.bw)
            .store("reshape", reshape_code, Reshape.fw, Reshape.bw)
            .store("transp", transp_code, Transpose.fw, Transpose.bw)
            .store("sum", sum_code, Sum.fw, Sum.bw)
            .store("conv2d", conv2d_code, Conv2D.fw, Conv2D.bw)
            .store("mpool2d", mpool2dd_code, MaxPool2D.fw, MaxPool2D.bw)
            .store("elu", elu_code, Elu.fw, Elu.bw)
            .store("exp", exp_code, Exp.fw, Exp.bw)
            .store("gelu", gelu_code, Gelu.fw, Gelu.bw)
            .store("h_sig", h_sig_code, HardSigmoid.fw, HardSigmoid.bw)
            .store("linear", linear_code, Linear.fw, Linear.bw)
            .store("mish", mish_code, Mish.fw, Mish.bw)
            .store("relu", relu_code, ReLu.fw, ReLu.bw)
            .store("selu", selu_code, Selu.fw, Selu.bw)
            .store("sig", sig_code, Sigmoid.fw, Sigmoid.bw)
            .store("softmax", softmax_code, Softmax.fw, Softmax.bw)
            .store("softplus", softplus_code, Softplus.fw, Softplus.bw)
            .store("softsign", softsign_code, Softsign.fw, Softsign.bw)
            .store("swish", swish_code, Swish.fw, Swish.bw)
            .store("tanh", tanh_code, Tanh.fw, Tanh.bw)
            .store("lrelu", lrelu_code, LeakyReLu.fw, LeakyReLu.bw)
            .store("dropout", dropout_code, Dropout.fw, Dropout.bw)
            .store("mae", mae_code, MAE.fw, MAE.bw)
            .store("mape", mape_code, MAPE.fw, MAPE.bw)
            .store("mse", mse_code, MSE.fw, MSE.bw)
            .store("msle", msle_code, MSLE.fw, MSLE.bw)
            .store("bce", bce_code, BCE.fw, BCE.bw)
            .store("cce", cce_code, CCE.fw, CCE.bw)
            .store("cfce", cfce_code, CFCE.fw, CFCE.bw)
        return Kernels {kernels: kernels}