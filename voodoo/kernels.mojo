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

    fn store(self, name: String, opcode: Int, fw: unary_op, bw: unary_op):
        self.kernels.store(opcode, op_tuple(name, fw, _b))
        self.kernels.store(opcode + 1, op_tuple(name + "_bw", bw, _b))

    fn store(self, name: String, opcode: Int, fw: binary_op, bw: binary_op):
        self.kernels.store(opcode, op_tuple(name, _u, fw))
        self.kernels.store(opcode + 1, op_tuple(name + "_bw", _u, bw))


@register_passable("trivial")
struct Kernels:
    var kernels: Pointer[op_tuple]

    fn __init__() -> Kernels:
        let kernels = Pointer[op_tuple].alloc(120)
        let k = KernelManager(kernels)
        k.store("cos", cos_code, Cos.fw, Cos.bw)
        k.store("sin", sin_code, Sin.fw, Sin.bw)
        k.store("tan", tan_code, Tan.fw, Tan.bw)
        k.store("acos", acos_code, Acos.fw, Acos.bw)
        k.store("asin", asin_code, Asin.fw, Asin.bw)
        k.store("atan", atan_code, Atan.fw, Atan.bw)
        k.store("cosh", cosh_code, Cosh.fw, Cosh.bw)
        k.store("sinh", sinh_code, Sinh.fw, Sinh.bw)
        k.store("log", log_code, Log.fw, Log.bw)
        k.store("log2", log2_code, Log2.fw, Log2.bw)
        k.store("exp2", exp2_code, Exp2.fw, Exp2.bw)
        k.store("sqrt", sqrt_code, Sqrt.fw, Sqrt.bw)
        k.store("abs", abs_code, Abs.fw, Abs.bw)
        k.store("copy", copy_code, Copy.fw, Copy.bw)
        k.store("add", add_code, Add.fw, Add.bw)
        k.store("sub", sub_code, Sub.fw, Sub.bw)
        k.store("mul", mul_code, Mul.fw, Mul.bw)
        k.store("div", div_code, Div.fw, Div.bw)
        k.store("pow", pow_code, Pow.fw, Pow.bw)
        k.store("mmul", mmul_code, MMul.fw, MMul.bw)
        k.store("reshape", reshape_code, Reshape.fw, Reshape.bw)
        k.store("transp", transp_code, Transpose.fw, Transpose.bw)
        k.store("sum", sum_code, Sum.fw, Sum.bw)
        k.store("conv2d", conv2d_code, Conv2D.fw, Conv2D.bw)
        k.store("mpool2d", mpool2dd_code, MaxPool2D.fw, MaxPool2D.bw)
        k.store("elu", elu_code, Exp.fw, Exp.bw)
        k.store("exp", exp_code, Exp.fw, Exp.bw)
        k.store("gelu", gelu_code, Gelu[0.0].fw, Gelu[0.0].bw)
        k.store("h_sig", hard_sigmoid_code, HardSigmoid.fw, HardSigmoid.bw)
        k.store("linear", linear_code, Linear.fw, Linear.bw)
        k.store("mish", mish_code, Mish.fw, Mish.bw)
        k.store("relu", relu_code, Relu[0.0, f32_max, 0.0].fw, Relu[0.0, f32_max, 0.0].bw)
        k.store("selu", selu_code, Selu.fw, Selu.bw)
        k.store("sig", sigmoid_code, Sigmoid.fw, Sigmoid.bw)
        k.store("softmax", softmax_code, Softmax.fw, Softmax.bw)
        k.store("softplus", softplus_code, Softplus.fw, Softplus.bw)
        k.store("softsign", softsign_code, Softsign.fw, Softsign.bw)
        k.store("swish", silu_code, Swish.fw, Swish.bw)
        k.store("tanh", tanh_code, Tanh.fw, Tanh.bw)
        k.store("lrelu", leaky_relu_code, LeakyRelu[0.0].fw, LeakyRelu[0.0].bw)
        k.store("dropout", dropout_code, Dropout.fw, Dropout.bw)
        k.store("mae", mae_code, MAE.fw, MAE.bw)
        k.store("mape", mape_code, MAPE.fw, MAPE.bw)
        k.store("mse", mse_code, MSE.fw, MSE.bw)
        k.store("msle", msle_code, MSLE.fw, MSLE.bw)
        k.store("bce", bce_code, BCE.fw, BCE.bw)
        k.store("cce", cce_code, CCE.fw, CCE.bw)
        k.store("cfce", cfce_code, CFCE.fw, CFCE.bw)
        return Kernels {kernels: kernels}
