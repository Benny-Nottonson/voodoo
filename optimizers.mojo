from utilities import Matrix
from math import sqrt, reciprocal

trait Optimizer(Copyable):
    fn apply(inout self):
        ...

@value
struct SGD(Optimizer):
    var lr: Float32
    var dW: Float32
    var db: Float32
    var W: Float32
    var b: Float32
    
    fn __init__(inout self, lr: Float32):
        self.lr = lr
        self.dW = 0
        self.db = 0
        self.W = 0
        self.b = 0

    fn apply(inout self):
        self.W += -self.lr * self.dW
        self.b += -self.lr * self.db

    fn zero_grad(inout self):
        self.dW = 0
        self.db = 0

    fn step(inout self):
        self.apply()
        self.zero_grad()


@value
struct Momentum(Optimizer):
    var lr: Float32
    var dW: Float32
    var db: Float32
    var W: Float32
    var b: Float32
    var vW: Float32
    var vb: Float32
    var beta: Float32

    fn __init__(inout self, lr: Float32, beta: Float32):
        self.lr = lr
        self.beta = beta
        self.dW = 0
        self.db = 0
        self.W = 0
        self.b = 0
        self.vW = 0
        self.vb = 0

    fn apply(inout self):
        self.vW = self.beta * self.vW + (1 - self.beta) * self.dW
        self.vb = self.beta * self.vb + (1 - self.beta) * self.db
        self.W += -self.lr * self.vW
        self.b += -self.lr * self.vb


@value
struct RMSProp(Optimizer):
    var lr: Float32
    var dW: Float32
    var db: Float32
    var W: Float32
    var b: Float32
    var sW: Float32
    var sb: Float32
    var beta: Float32
    var eps: Float32

    fn __init__(inout self, lr: Float32, beta: Float32, eps: Float32):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.dW = 0
        self.db = 0
        self.W = 0
        self.b = 0
        self.sW = 0
        self.sb = 0

    fn apply(inout self):
        self.sW = self.beta * self.sW + (1 - self.beta) * self.dW * self.dW
        self.sb = self.beta * self.sb + (1 - self.beta) * self.db * self.db
        self.W += -self.lr * self.dW * reciprocal(sqrt(self.sW + self.eps))
        self.b += -self.lr * self.db * reciprocal(sqrt(self.sb + self.eps))


@value 
struct AdaGrad(Optimizer):
    var lr: Float32
    var dW: Float32
    var db: Float32
    var W: Float32
    var b: Float32
    var sW: Float32
    var sb: Float32
    var eps: Float32

    fn __init__(inout self, lr: Float32, eps: Float32):
        self.lr = lr
        self.eps = eps
        self.dW = 0
        self.db = 0
        self.W = 0
        self.b = 0
        self.sW = 0
        self.sb = 0

    fn apply(inout self):
        self.sW += self.dW * self.dW
        self.sb += self.db * self.db
        self.W += -self.lr * self.dW * reciprocal(sqrt(self.sW + self.eps))
        self.b += -self.lr * self.db * reciprocal(sqrt(self.sb + self.eps))


@value
struct Adam(Optimizer):
    var lr: Float32
    var dW: Float32
    var db: Float32
    var W: Float32
    var b: Float32
    var vW: Float32
    var vb: Float32
    var sW: Float32
    var sb: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var t: Int32

    fn __init__(inout self, lr: Float32, beta1: Float32, beta2: Float32, eps: Float32):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dW = 0
        self.db = 0
        self.W = 0
        self.b = 0
        self.vW = 0
        self.vb = 0
        self.sW = 0
        self.sb = 0
        self.t = 0

    fn apply(inout self):
        self.t += 1
        self.vW = self.beta1 * self.vW + (1 - self.beta1) * self.dW
        self.vb = self.beta1 * self.vb + (1 - self.beta1) * self.db
        self.sW = self.beta2 * self.sW + (1 - self.beta2) * self.dW * self.dW
        self.sb = self.beta2 * self.sb + (1 - self.beta2) * self.db * self.db
        self.W += -self.lr * self.vW * reciprocal(sqrt(self.sW + self.eps))
        self.b += -self.lr * self.vb * reciprocal(sqrt(self.sb + self.eps))