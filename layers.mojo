from utilities import Matrix
from activation import ActivationLayer, ReLU
from initializer import Initializer, XavierUniform, Constant
from random import random_float64

@value
struct HiddenLayers:
    var layers: VariadicList[ParamLayer]

    fn __init__(inout self, layers: VariadicList[ParamLayer]):
        self.layers = layers
        
trait ParamLayer(Copyable):
    fn backward(inout self, delta: Matrix) -> Matrix:
        ...

    fn __call__(inout self, x: Matrix) -> Matrix:
        ...


@value
struct Dense(ParamLayer):
    var W: Matrix
    var b: Matrix
    var dW: Matrix
    var db: Matrix
    var z: Matrix
    var input: Matrix
    var regularizer_type: String
    var lam: Float32
    var in_features: Int
    var out_features: Int
    var act: ActivationLayer
    var weight_initializer: Initializer
    var bias_initializer: Initializer

    fn __init__(
        inout self,
        in_features: Int,
        out_features: Int,
        activation: ActivationLayer = ReLU,
        weight_initializer: Initializer = XavierUniform,
        bias_initializer: Initializer = Constant,
        regularizer_type: String = "",
        lam: Float32 = 0.0,
    ):
        let i = in_features
        let j = out_features
        let init_weight = Matrix(i, j)
        let init_bias = Matrix(1, j)
        self.W = weight_initializer.initialize(init_weight)
        self.b = bias_initializer.initialize(init_bias)
        self.dW = Matrix(j, i)
        self.db = Matrix(1, j)
        self.z = Matrix(0, 0)
        self.input = Matrix(0, 0)
        self.regularizer_type = regularizer_type
        self.lam = lam
        self.in_features = in_features
        self.out_features = out_features
        self.act = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer_type = regularizer_type
        self.lam = lam

    fn forward(inout self, x: Matrix) -> Matrix:
        self.input = x
        let z = x @ self.W
        var b = self.b.deepcopy()
        while len(b) < len(z):
            b.append(self.b[0])
        self.z = z + b
        return self.act(self.z)

    fn backward(inout self, delta: Matrix) -> Matrix:
        let dz = self.act.derivative(self.z) * delta
        let dw_unscale = self.input.t() @ dz
        self.dW = dw_unscale * (1 / len(dz)).cast[DType.float32]()
        var ones_t = Matrix(1, len(dz))
        ones_t.fill(1)
        let db_unscale = ones_t @ dz
        self.db = db_unscale * (1 / len(dz)).cast[DType.float32]()

        if self.regularizer_type == "l2":
            self.dW = self.dW + (self.W * self.lam)
            self.db = self.db + (self.b * self.lam)
        elif self.regularizer_type == "l1":
            self.dW = self.dW + self.lam
            self.db = self.db + self.lam
        return dz @ self.W.t()

    fn __call__(inout self, x: Matrix) -> Matrix:
        return self.forward(x)


@value
struct BatchNorm1d(ParamLayer):
    var W: Matrix
    var b: Matrix
    var dW: Matrix
    var db: Matrix
    var z: Matrix
    var input: Matrix
    var regularizer_type: String
    var lam: Float32
    var in_features: Int
    var x_hat: Matrix
    var eps: Float32
    var beta: Float32
    var mu: Matrix
    var std: Matrix
    var mu_hat: Matrix
    var std_hat: Matrix
    var gamma: Matrix

    fn __init__(inout self, in_features: Int):
        self.W = Constant(1.0).initialize(Matrix(1, in_features))
        self.b = Constant(0.0).initialize(Matrix(1, in_features))
        self.dW = Matrix(1, in_features)
        self.db = Matrix(1, in_features)
        self.z = Matrix(0, 0)
        self.input = Matrix(0, 0)
        self.regularizer_type = ""
        self.lam = 0.0
        self.in_features = in_features
        self.x_hat = Matrix(0, 0)
        self.eps = 1e-5
        self.beta = 0.9
        self.mu = Matrix(1, in_features)
        self.std = Matrix(1, in_features)
        self.mu_hat = Matrix(1, in_features)
        self.std_hat = Matrix(1, in_features)
        self.std_hat.fill(1)
        self.gamma = Matrix(0, 0)

    fn forward(inout self, x: Matrix, eval: Bool) -> Matrix:
        if not eval:
            self.mu.fill(x.mean())
            self.std.fill(x.variance() ** 0.5)
            self.mu_hat = self.mu_hat * (1 - self.beta) + (self.mu * self.beta)
            self.std_hat = self.std_hat * (1 - self.beta) + (self.std * self.beta)
        else:
            self.mu = self.mu_hat
            self.std = self.std_hat
        var mu = self.mu.deepcopy()
        var std = self.std.deepcopy()
        while len(mu) < len(x):
            mu.append(self.mu[0])
            std.append(self.std[0])
        let num = x + (mu * -1)
        let den = ((std * std) + self.eps) ** 0.5
        let x_hat = num * (den**-1)
        self.x_hat = x_hat

        self.gamma = self.W.deepcopy()
        var beta = self.b.deepcopy()
        while len(self.gamma) < len(x):
            self.gamma.append(self.W[0])
            beta.append(self.b[0])

        return (self.gamma * x_hat) + beta

    fn backward(inout self, delta: Matrix) -> Matrix:
        let dz = delta
        let dx_hat = dz * self.gamma
        let m = len(dz)
        self.dW.fill((self.x_hat * dz).sum() * (1 / m).cast[DType.float32]())
        self.db.fill(dz.sum() * (1 / m).cast[DType.float32]())

        let a1 = dx_hat * m
        let a2 = dx_hat.sum()
        let a3 = self.x_hat * (dx_hat * self.x_hat).sum()
        let num = a1 + (a2 * -1) + (a3 * -1)
        let den = (((self.std * self.std) + self.eps) ** 0.5) * m

        return num * (den**-1)

    fn __call__(inout self, x: Matrix) -> Matrix:
        return self.forward(x, False)


@value
struct Dropout(ParamLayer):
    var p: Float32
    var mask: Matrix
    var input: Matrix

    fn __init__(inout self, p: Float32):
        self.p = p
        self.mask = Matrix(0, 0)
        self.input = Matrix(0, 0)

    fn forward(inout self, x: Matrix, eval: Bool) -> Matrix:
        if not eval:
            self.mask = Matrix(x.shape.get[0, Int](), x.shape.get[1, Int]())
            for i in range(self.mask.rows):
                for j in range(self.mask.cols):
                    if random_float64(0, 1).cast[DType.float32]() < self.p:
                        self.mask[i, j] = 0
                    else:
                        self.mask[i, j] = 1
            self.input = x
            return x * self.mask
        else:
            return x * (1 - self.p)

    fn backward(inout self, delta: Matrix) -> Matrix:
        return delta * self.mask

    fn __call__(inout self, x: Matrix) -> Matrix:
        return self.forward(x, False)
