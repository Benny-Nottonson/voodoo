from math import log
from utilities import Matrix


@value
struct Loss:
    var value: Float32
    var delta: Matrix

    fn __str__(self) -> String:
        return "Loss: " + str(self.value)

    fn backward(inout self):
        self.delta = self.delta * self.value


trait LossFunc(Copyable):
    fn delta(self) -> Matrix:
        ...

    fn __call__(inout self, pred: Matrix, target: Matrix) -> Loss:
        ...


@value
struct MSELoss(LossFunc):
    var pred: Matrix
    var target: Matrix

    fn __init__(inout self):
        self.pred = Matrix(0, 0)
        self.target = Matrix(0, 0)

    fn apply(inout self, p: Matrix, t: Matrix) -> Loss:
        self.pred = p
        self.target = t
        var loss: SIMD[DType.float32, 1] = 0
        for i in range(len(p)):
            loss += (p[i][0] - t[i][0]) ** 2
        return Loss(loss / len(p), self.delta())

    fn delta(self) -> Matrix:
        return self.pred + (self.target * -1)

    fn __call__(inout self, pred: Matrix, target: Matrix) -> Loss:
        return self.apply(pred, target)


@value
struct CrossEntropyLoss(LossFunc):
    var pred: Matrix
    var target: Matrix

    fn __init__(inout self):
        self.pred = Matrix(0, 0)
        self.target = Matrix(0, 0)

    fn apply(inout self, p: Matrix, t: Matrix) -> Loss:
        self.pred = p
        self.target = t
        var loss: SIMD[DType.float32, 1] = 0
        for i in range(len(p)):
            let firstIndex = (t[i][0]).to_int()
            let el = p[i][firstIndex]
            loss += -log(el + 1e-6)
        return Loss(loss / len(p), self.delta())

    fn delta(self) -> Matrix:
        var probs = self.soft_max(self.pred)
        let w = len(self.pred)
        for i in range(w):
            let firstIndex = (self.target[i][0]).to_int()
            probs[i][firstIndex] -= 1
        return probs

    fn soft_max(self, x: Matrix) -> Matrix:
        let w = len(x)
        let h = len(x[0])
        var num = Matrix(w, h)
        var den = Matrix(w, 1)
        for i in range(w):
            var max_of_batch = x[i][0]
            var sum_of_batch: SIMD[DType.float32, 1] = 0
            for j in range(h):
                if x[i][j] > max_of_batch:
                    max_of_batch = x[i][j]
            for j in range(h):
                num[i][j] = math.exp(x[i][j] - max_of_batch)
                sum_of_batch += num[i][j]
            den[i][0] = sum_of_batch

        for i in range(w):
            for j in range(h):
                num[i][j] = num[i][j] / den[i][0] + 1e-6
        return num

    fn __call__(inout self, pred: Matrix, target: Matrix) -> Loss:
        return self.apply(pred, target)
