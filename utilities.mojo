from memory import memset_zero
from random import rand
from math import log, abs


fn broadcast(
    a: Matrix,
    b: Matrix,
) -> Tuple[Matrix, Matrix]:
    var newA = Matrix(a.rows, b.cols)
    var newB = Matrix(a.rows, b.cols)

    for y in range(a.rows):
        for x in range(b.cols):
            newA[y, x] = a[y, 0]
            newB[y, x] = b[y, x]

    return newA, newB


fn naive_matmul(
    a: Matrix,
    b: Matrix,
) -> Matrix:
    var new = Matrix(a.rows, b.cols)

    for y in range(a.rows):
        for x in range(b.cols):
            var sum: SIMD[MatrixType, 1] = 0
            for i in range(a.cols):
                sum += a[y, i] * b[i, x]
            new[y, x] = sum

    return new


fn strassen_matmul(
    a: Matrix,
    b: Matrix,
) -> Matrix:
    if a.rows <= 64:
        return naive_matmul(a, b)

    var new = Matrix(a.rows, b.cols)

    let half = a.rows // 2

    let a11 = a.slice(0, 0, half, half)
    let a12 = a.slice(0, half, half, half)
    let a21 = a.slice(half, 0, half, half)
    let a22 = a.slice(half, half, half, half)

    let b11 = b.slice(0, 0, half, half)
    let b12 = b.slice(0, half, half, half)
    let b21 = b.slice(half, 0, half, half)
    let b22 = b.slice(half, half, half, half)

    let m1 = strassen_matmul(a11 + a22, b11 + b22)
    let m2 = strassen_matmul(a21 + a22, b11)
    let m3 = strassen_matmul(a11, b12 - b22)
    let m4 = strassen_matmul(a22, b21 - b11)

    let m5 = strassen_matmul(a11 + a12, b22)
    let m6 = strassen_matmul(a21 - a11, b11 + b12)
    let m7 = strassen_matmul(a12 - a22, b21 + b22)

    let c11 = m1 + m4 - m5 + m7
    let c12 = m3 + m5
    let c21 = m2 + m4
    let c22 = m1 - m2 + m3 + m6

    for y in range(half):
        for x in range(half):
            new[y, x] = c11[y, x]
            new[y, x + half] = c12[y, x]
            new[y + half, x] = c21[y, x]
            new[y + half, x + half] = c22[y, x]

    return new


alias MatrixType = DType.float32


struct Matrix(Stringable, Sized):
    var rows: Int
    var cols: Int
    var shape: Tuple[Int, Int]
    var data: DTypePointer[MatrixType]

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[MatrixType].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)

    fn __init__(inout self, shape: Tuple[Int, Int]):
        let rows = shape.get[0, Int]()
        let cols = shape.get[1, Int]()
        self.data = DTypePointer[MatrixType].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols
        self.shape = shape

    fn __init__(inout self, shape: Tuple[Int, Int], init: Float32):
        let rows = shape.get[0, Int]()
        let cols = shape.get[1, Int]()
        self.data = DTypePointer[MatrixType].alloc(rows * cols)
        for i in range(rows * cols):
            self.data[i] = init
        self.rows = rows
        self.cols = cols
        self.shape = shape


    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[MatrixType]):
        self.data = data
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)

    @staticmethod
    fn log(m: Matrix) -> Matrix:
        var new = Matrix(m.rows, m.cols)
        for y in range(m.rows):
            for x in range(m.cols):
                new[y, x] = log(m[y, x])
        return new

    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[MatrixType].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[MatrixType, 1]:
        return self.load(y, x)

    fn __getitem__(self, y: Int) -> SIMD[MatrixType]:
        var row = SIMD[MatrixType](self.cols)
        for x in range(self.cols):
            row[x] = self[y, x]
        return row

    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[MatrixType, 1]):
        self.store(y, x, val)

    fn __setitem__[cols: Int](inout self, y: Int, val: SIMD[MatrixType, cols]):
        for x in range(cols):
            self.store(y, x, val[x])

    fn __copyinit__(inout self, other: Self):
        self.data = other.data
        self.rows = other.rows
        self.cols = other.cols
        self.shape = other.shape

    fn __moveinit__(inout self, owned other: Self):
        self.data = other.data
        self.rows = other.rows
        self.cols = other.cols
        self.shape = other.shape

    fn slice(self, y: Int, x: Int, rows: Int, cols: Int) -> Self:
        var new = Self(rows, cols)
        for i in range(rows):
            for j in range(cols):
                new[i, j] = self[y + i, x + j]
        return new

    fn deepcopy(borrowed self) -> Self:
        var new = Self(self.rows, self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x]
        return new

    fn fill(inout self, val: SIMD[MatrixType, 1]):
        for y in range(self.rows):
            for x in range(self.cols):
                self[y, x] = val

    fn load(self, y: Int, x: Int) -> SIMD[MatrixType, 1]:
        return self.data.simd_load[1](y * self.cols + x)

    fn store(self, y: Int, x: Int, val: SIMD[MatrixType, 1]):
        return self.data.simd_store[1](y * self.cols + x, val)

    fn __len__(self) -> Int:
        return self.rows

    fn __matmul__(self, other: Self) -> Self:
        return strassen_matmul(self, other)

    fn __mul__(self, other: Self) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] * other[y, x]

        return new

    fn __mul__(self, other: SIMD[MatrixType, 1]) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] * other

        return new

    fn __add__(self, other: Self) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] + other[y, x]

        return new

    fn __add__(self, other: SIMD[MatrixType, 1]) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] + other

        return new

    fn __pow__(self, other: Float32) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] ** other

        return new

    fn __sub__(self, other: Self) -> Self:
        return self.__add__(other * -1)

    fn __neg__(self) -> Self:
        return self * -1

    fn __truediv__(self, other: Self) -> Self:
        var new = Self(self.rows, self.cols)

        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] / other[y, x]

        return new

    fn __repr__(self) -> String:
        var s = String("[\n")
        for y in range(self.rows):
            s += String("[")
            for x in range(self.cols):
                s += String(self[y, x])
                if x != self.cols - 1:
                    s += String(", ")
            s += String("]")
            if y != self.rows - 1:
                s += String(",\n")
        s += String("\n]\n")
        return s

    fn __str__(self) -> String:
        return self.__repr__()

    fn append(inout self, row: SIMD[MatrixType]):
        self.data.simd_store(self.rows * self.cols, row)
        self.rows += 1
        self._update_shape()

    fn _update_shape(inout self):
        self.shape = (self.rows, self.cols)

    fn sum(self) -> SIMD[MatrixType, 1]:
        var sum: SIMD[MatrixType, 1] = 0
        for y in range(self.rows):
            for x in range(self.cols):
                sum += self[y, x]
        return sum

    fn t(self) -> Self:
        var new = Self(self.cols, self.rows)
        for y in range(self.rows):
            for x in range(self.cols):
                new[x, y] = self[y, x]
        return new

    fn mean(self) -> SIMD[MatrixType, 1]:
        return self.sum() / (self.rows * self.cols)

    fn square(self) -> Self:
        var new = Self(self.rows, self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x] ** 2
        return new

    fn abs(self) -> Self:
        var new = Self(self.rows, self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = abs(self[y, x])
        return new

    fn variance(self) -> SIMD[MatrixType, 1]:
        let mean = self.mean()
        var variance: SIMD[MatrixType, 1] = 0
        for y in range(self.rows):
            for x in range(self.cols):
                variance += (self[y, x] - mean) ** 2
        return variance / (self.rows * self.cols)

    fn concat(self, other: Self) -> Self:
        var new = Self(self.rows + other.rows, self.cols)
        for y in range(self.rows):
            new[self.cols, y] = self[self.cols, y]
        for y in range(other.rows):
            new[self.cols, y + self.rows] = other[self.cols, y]
        return new

    fn stack(self, other: Self) -> Self:
        var new = Self(self.rows, self.cols + other.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                new[y, x] = self[y, x]
        for y in range(other.rows):
            for x in range(other.cols):
                new[y, x + self.cols] = other[y, x]
        return new
