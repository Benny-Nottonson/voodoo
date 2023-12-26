from math import tanh, exp
from utilities import Matrix


trait ActivationLayer(Copyable):
    fn __call__(self, x: Matrix) -> Matrix:
        ...

    fn forward(self, x: Matrix) -> Matrix:
        ...

    fn derivative(self, x: Matrix) -> Matrix:
        ...


@value
struct Linear(ActivationLayer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    fn forward(self, x: Matrix) -> Matrix:
        return x

    fn derivative(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)

        for i in range(w):
            for j in range(h):
                temp[i, j] = 1
        return temp


@value
struct ReLU(ActivationLayer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    fn forward(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = x[i, j] if x[i, j] > 0 else 0
        return temp

    fn derivative(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = 1 if x[i, j] > 0 else 0
        return temp


@value
struct Tanh(ActivationLayer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    fn forward(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = tanh(x[i, j])
        return temp

    fn derivative(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        let temp = self.forward(x)
        var temp2 = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp2[i, j] = 1 - temp[i, j] ** 2
        return temp2


@value
struct Sigmoid(ActivationLayer):
    fn __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    fn forward(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        var temp = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp[i, j] = 1 / (1 + exp(-x[i, j]))
        return temp

    fn derivative(self, x: Matrix) -> Matrix:
        let w = x.rows
        let h = x.cols
        let temp = self.forward(x)
        var temp2 = Matrix(w, h)
        for i in range(w):
            for j in range(h):
                temp2[i, j] = temp[i, j] * (1 - temp[i, j])
        return temp2
