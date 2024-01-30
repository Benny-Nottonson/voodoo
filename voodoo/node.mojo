from math import sin, cos, sqrt, log, iota
from random import rand, seed
from .utils import Vector, warn

# This means it can never be passed as a reference, only a value
# Because of this, anything mutable inside the struct needs to be a pointer
# This does not include Vector as it is register passable


@register_passable("trivial")
struct Node:
    var id_ptr: Pointer[Int]
    var data_id_ptr: Pointer[Int]
    var grad_id_ptr: Pointer[Int]
    var data_ptr: Pointer[DTypePointer[DType.float32]]
    var parents: Vector[Int]
    var children: Vector[Int]
    var dependencies_ptr: Pointer[Int]
    var is_static_ptr: Pointer[Bool]
    var computed_ptr: Pointer[Bool]
    var grad_computed_ptr: Pointer[Bool]
    var operator_id_ptr: Pointer[Int]
    var grad_operator_id_ptr: Pointer[Int]
    var requires_grad_ptr: Pointer[Bool]
    var tmp_visited_ptr: Pointer[Bool]
    var checkpoint_ptr: Pointer[Bool]
    var is_single_ptr: Pointer[Bool]
    var cap_ptr: Pointer[Int]
    var num_dims_ptr: Pointer[Int]
    var shape: Vector[Int]
    var strides: Vector[Int]
    var other_params: Vector[Int]

    fn __init__(
        id: Int,
        shape: Vector[Int],
        is_static: Bool = True,
        other_params: Vector[Int] = Vector[Int](),
    ) -> Self:
        let id_ptr = Pointer[Int].alloc(1)
        id_ptr.store(id)
        let data_id_ptr = Pointer[Int].alloc(1)
        data_id_ptr.store(-1)
        let grad_id_ptr = Pointer[Int].alloc(1)
        grad_id_ptr.store(-1)
        let data_ptr = Pointer[DTypePointer[DType.float32]].alloc(2)
        let data = DTypePointer[DType.float32].get_null()
        let grad = DTypePointer[DType.float32].get_null()
        data_ptr.store(0, data)
        data_ptr.store(1, grad)
        let parents = Vector[Int]()
        let children = Vector[Int]()
        let dependencies_ptr = Pointer[Int].alloc(1)
        dependencies_ptr.store(0)
        let is_static_ptr = Pointer[Bool].alloc(1)
        is_static_ptr.store(is_static)
        let computed_ptr = Pointer[Bool].alloc(1)
        computed_ptr.store(is_static)
        let grad_computed_ptr = Pointer[Bool].alloc(1)
        grad_computed_ptr.store(False)
        let operator_id_ptr = Pointer[Int].alloc(1)
        operator_id_ptr.store(-1)
        let grad_operator_id_ptr = Pointer[Int].alloc(1)
        grad_operator_id_ptr.store(-1)
        let requires_grad_ptr = Pointer[Bool].alloc(1)
        requires_grad_ptr.store(is_static)
        let tmp_visited_ptr = Pointer[Bool].alloc(1)
        tmp_visited_ptr.store(False)
        let checkpoint_ptr = Pointer[Bool].alloc(1)
        checkpoint_ptr.store(False)
        let is_single_ptr = Pointer[Bool].alloc(1)
        is_single_ptr.store(False)
        let num_dims_ptr = Pointer[Int].alloc(1)
        num_dims_ptr.store(shape.len.load())
        let cap_ptr = Pointer[Int].alloc(1)
        cap_ptr.store(1)

        for i in range(shape.len.load()):
            cap_ptr.store(cap_ptr.load() * shape.load(i))

        let strides = Vector[Int](shape.len.load())
        strides.store(shape.len.load() - 1, 1)
        for i in range(shape.len.load() - 1):
            strides.store(
                shape.len.load() - i - 2,
                strides.load(shape.len.load() - i - 1)
                * shape.load(shape.len.load() - i - 1),
            )

        return Node {
            id_ptr: id_ptr,
            data_id_ptr: data_id_ptr,
            grad_id_ptr: grad_id_ptr,
            data_ptr: data_ptr,
            parents: parents,
            children: children,
            dependencies_ptr: dependencies_ptr,
            is_static_ptr: is_static_ptr,
            computed_ptr: computed_ptr,
            grad_computed_ptr: grad_computed_ptr,
            operator_id_ptr: operator_id_ptr,
            grad_operator_id_ptr: grad_operator_id_ptr,
            requires_grad_ptr: requires_grad_ptr,
            tmp_visited_ptr: tmp_visited_ptr,
            checkpoint_ptr: checkpoint_ptr,
            is_single_ptr: is_single_ptr,
            cap_ptr: cap_ptr,
            num_dims_ptr: num_dims_ptr,
            shape: shape,
            strides: strides,
            other_params: other_params,
        }

    @always_inline("nodebug")
    fn is_zero(self) -> Bool:
        for i in range(self.cap_ptr.load()):
            if self.data_ptr.load(0).load(i) != 0.0:
                return False
        return True

    @always_inline("nodebug")
    fn fill(self, val: Float32):
        for i in range(self.cap_ptr.load()):
            self.data_ptr.load(0).store(i, val)

    @always_inline("nodebug")
    fn fill_incr(self):
        iota(self.data_ptr.load(0), self.cap_ptr.load())

    @always_inline("nodebug")
    fn fill_grad(self, val: Float32):
        for i in range(self.cap_ptr.load()):
            self.data_ptr.load(1).store(i, val)

    @always_inline("nodebug")
    fn grad_fill_incr(self):
        iota(self.data_ptr.load(1), self.cap_ptr.load())

    fn initialize[
        initialization_function: String, val: Float32 = 0, val2: Float32 = 0
    ](self):
        @parameter
        if initialization_function == "glorot_normal":
            self.glorot_normal()
        elif initialization_function == "glorot_uniform":
            self.glorot_uniform()
        elif initialization_function == "he_normal":
            self.he_normal()
        elif initialization_function == "he_uniform":
            self.he_uniform()
        elif initialization_function == "identity":
            self.identity()
        elif initialization_function == "lecun_normal":
            self.lecun_normal()
        elif initialization_function == "lecun_uniform":
            self.lecun_uniform()
        elif initialization_function == "ones":
            self.ones()
        elif initialization_function == "random_normal":
            self.random_normal()
        elif initialization_function == "random_uniform":
            self.random_uniform(val, val2)
        elif initialization_function == "truncated_normal":
            self.truncated_normal()
        elif initialization_function == "zeros":
            self.zeros()
        elif initialization_function == "fill":
            self.fill(val)
        elif initialization_function == "fill_incr":
            self.fill_incr()
        elif initialization_function == "grad_fill_incr":
            self.grad_fill_incr()
        else:
            warn(
                "Invalid initialization function: "
                + initialization_function
                + " using zeros\n"
            )
            self.zeros()

    fn glorot_normal(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let fan_out = self.shape.load(self.shape.len.load() - 1)
        let scale = sqrt(2.0 / Float32(fan_in + fan_out))
        self.random_normal(scale, 0.0)

    fn glorot_uniform(self):
        let fan_in: Float32 = self.shape.load(self.shape.len.load() - 2)
        let fan_out: Float32 = self.shape.load(self.shape.len.load() - 1)
        let scale = sqrt(6.0 / (fan_in + fan_out))
        self.random_uniform(-scale, scale)

    fn he_normal(self):
        let fan_in: Float32 = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(2.0 / fan_in)
        self.random_normal(scale, 0.0)

    fn he_uniform(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(6.0 / Float32(fan_in))
        self.random_uniform(-scale, scale)

    fn he_random(self):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        let u2 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        rand(u1, self.cap_ptr.load())
        rand(u2, self.cap_ptr.load())
        for i in range(self.cap_ptr.load()):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            let sigma = sqrt(2.0 / Float32(self.shape.load(self.shape.len.load() - 1)))
            self.data_ptr.load(0).store(i, z * sigma)

    fn identity(self):
        let row: Int = self.shape.load(self.num_dims_ptr.load() - 2)
        let cols: Int = self.shape.load(self.num_dims_ptr.load() - 1)
        let col_strides: Int = (self.strides.load(0) * self.shape.load(0)) // cols
        for i in range(col_strides):
            for j in range(cols):
                if i == j:
                    self.data_ptr.load(0).store(i * cols + j, 1.0)
                else:
                    self.data_ptr.load(0).store(i * cols + j, 0.0)

    fn lecun_normal(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(1.0 / Float32(fan_in))
        self.random_normal(scale, 0.0)

    fn lecun_uniform(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(3.0 / Float32(fan_in))
        self.random_uniform(-scale, scale)

    fn ones(self):
        self.fill(1.0)

    fn random_normal(self, std: Float32 = 1.0, mu: Float32 = 0.0):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        let u2 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        rand(u1, self.cap_ptr.load())
        rand(u2, self.cap_ptr.load())
        for i in range(self.cap_ptr.load()):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            self.data_ptr.load(0).store(i, z * std + mu)

    fn random_uniform(self, min: Float32, max: Float32):
        seed()
        rand(self.data_ptr.load(0), self.cap_ptr.load())
        for i in range(self.cap_ptr.load()):
            self.data_ptr.load(0).store(i, self.data_ptr.load(0).load(i) * (max - min) + min)

    fn truncated_normal(self, std: Float32 = 1.0, mu: Float32 = 0.0):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        let u2 = DTypePointer[DType.float32].alloc(self.cap_ptr.load())
        rand(u1, self.cap_ptr.load())
        rand(u2, self.cap_ptr.load())
        for i in range(self.cap_ptr.load()):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            if z > -2.0 and z < 2.0:
                self.data_ptr.load(0).store(i, z * std + mu)
            else:
                self.data_ptr.load(0).store(i, 0.0)

    fn zeros(self):
        self.fill(0.0)

    fn orthoganal(self, gain: Float32 = 1.0):
        let row: Int = self.shape.load(self.num_dims_ptr.load() - 2)
        let cols: Int = self.shape.load(self.num_dims_ptr.load() - 1)
        let col_strides: Int = (self.strides.load(0) * self.shape.load(0)) // cols
        let tmp = DTypePointer[DType.float32](col_strides)
        for i in range(col_strides):
            for j in range(cols):
                if i == j:
                    tmp.store(i * cols + j, 1.0)
                else:
                    tmp.store(i * cols + j, 0.0)
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(col_strides)
        let u2 = DTypePointer[DType.float32].alloc(col_strides)
        rand(u1, col_strides)
        rand(u2, col_strides)
        for i in range(col_strides):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            tmp.store(i, z)
        let tmp2 = DTypePointer[DType.float32](col_strides)
        for i in range(col_strides):
            tmp2.store(i, tmp.load(i))
        for i in range(col_strides):
            for j in range(cols):
                tmp.store(i * cols + j, tmp2.load(i) * gain)
        for i in range(col_strides):
            for j in range(cols):
                self.data_ptr.load(0).store(i * cols + j, tmp.load(i * cols + j))

    fn free(self):
        self.id_ptr.free()
        self.data_id_ptr.free()
        self.grad_id_ptr.free()
        self.data_ptr.load(0).free()
        self.data_ptr.load(1).free()
        self.data_ptr.free()
        self.parents.free()
        self.children.free()
        self.dependencies_ptr.free()
        self.is_static_ptr.free()
        self.computed_ptr.free()
        self.grad_computed_ptr.free()
        self.operator_id_ptr.free()
        self.grad_operator_id_ptr.free()
        self.requires_grad_ptr.free()
        self.tmp_visited_ptr.free()
        self.checkpoint_ptr.free()
        self.is_single_ptr.free()
        self.cap_ptr.free()
        self.num_dims_ptr.free()
        self.shape.free()
        self.strides.free()
        self.other_params.free()

    fn print(self, accuracy: Int = 6):
        let row: Int = self.shape.load(self.num_dims_ptr.load() - 2)
        let cols: Int = self.shape.load(self.num_dims_ptr.load() - 1)
        let col_strides: Int = (self.strides.load(0) * self.shape.load(0)) // cols
        print(" ")
        var times = 1
        if self.grad_computed_ptr.load() and self.grad_id_ptr.load() != -1:
            times = 2
        print_no_newline("<Tensor: ")
        for i in range(col_strides):
            if col_strides > 10 and i > 4 and i < col_strides - 5:
                if i == 5:
                    print("                 ... ")
                continue
            else:
                if i > 0:
                    print_no_newline("           ")
                else:
                    print_no_newline("[ ")

                var indent = 0
                for d in range(self.num_dims_ptr.load() - 1):
                    if cols * i % self.strides.load(d) == 0:
                        print_no_newline("[ ")
                        indent += 1
                    else:
                        print_no_newline("  ")

                for j in range(cols):
                    if cols > 10 and j >= 3 and j < cols - 3:
                        if j == 3:
                            print_no_newline("... , ")
                        continue
                    else:
                        let idx = cols * i + j
                        print_no_newline(
                            String(self.data_ptr.load(0).load(idx))[
                                :accuracy
                            ] if self.data_ptr.load(0).load(idx)
                            != 0.0 else String(0.000)[:accuracy]
                        )
                        if j != cols - 1:
                            print_no_newline(", ")

                for d in range(self.num_dims_ptr.load() - 2, -1, -1):
                    if cols * (i + 1) % self.strides.load(d) == 0:
                        print_no_newline(" ]")

                if i < col_strides - 1:
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(self.num_dims_ptr.load()):
                        print_no_newline(self.shape.load(i))
                        if i < self.num_dims_ptr.load() - 1:
                            print_no_newline(",")
                    print_no_newline("] ")
                    print_no_newline(">")
        print(" ")
