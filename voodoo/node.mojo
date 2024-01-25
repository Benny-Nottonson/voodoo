from memory import memset_zero
from algorithm import vectorize
from math import round, ceil, sin, cos, sqrt, log
from random import rand, seed
from .utils import Vector, warn
from memory.buffer import Buffer


@register_passable("trivial")
struct Node:
    var id_ptr: Pointer[Int]
    var data_id: Pointer[Int]
    var grad_id: Pointer[Int]
    var data: Pointer[DTypePointer[DType.float32]]
    var parents_ptr: Pointer[Vector[Int]]
    var children_ptr: Pointer[Vector[Int]]
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
    var shape_ptr: Pointer[Vector[Int]]
    var strides_ptr: Pointer[Vector[Int]]
    var other_params_ptr: Pointer[Vector[Int]]

    fn __init__(id: Int, shape: Vector[Int], is_static: Bool = True) -> Self:
        let id_ptr = Pointer[Int].alloc(1)
        id_ptr.store(id)

        let data_id = Pointer[Int].alloc(1)
        data_id.store(-1)

        let grad_id = Pointer[Int].alloc(1)
        grad_id.store(-1)

        let data = Pointer[DTypePointer[DType.float32]].alloc(2)
        data.store(0, DTypePointer[DType.float32].get_null())
        data.store(1, DTypePointer[DType.float32].get_null())

        let parents_ptr = Pointer[Vector[Int]].alloc(1)
        parents_ptr.store(Vector[Int]())

        let children_ptr = Pointer[Vector[Int]].alloc(1)
        children_ptr.store(Vector[Int]())

        let dependencies_ptr = Pointer[Int].alloc(1)
        dependencies_ptr.store(0)

        let is_static_ptr = Pointer[Bool].alloc(1)
        is_static_ptr.store(is_static)

        let computed_ptr = Pointer[Bool].alloc(1)
        computed_ptr.store(is_static)

        let grad_computed_ptr = Pointer[Bool].alloc(1)
        grad_computed_ptr.store(False)

        let requires_grad_ptr = Pointer[Bool].alloc(1)
        requires_grad_ptr.store(is_static)

        let operator_id_ptr = Pointer[Int].alloc(1)
        operator_id_ptr.store(-1)

        let grad_operator_id_ptr = Pointer[Int].alloc(1)
        grad_operator_id_ptr.store(-1)

        let tmp_visited_ptr = Pointer[Bool].alloc(1)
        tmp_visited_ptr.store(False)

        let checkpoint_ptr = Pointer[Bool].alloc(1)
        checkpoint_ptr.store(False)

        let is_single_ptr = Pointer[Bool].alloc(1)
        is_single_ptr.store(False)

        let _num_dims = shape.len.load()
        let num_dims_ptr = Pointer[Int].alloc(1)
        num_dims_ptr.store(_num_dims)

        var _cap = shape.load(0)
        for i in range(1, _num_dims):
            _cap *= shape.load(i)
        let cap_ptr = Pointer[Int].alloc(1)
        cap_ptr.store(_cap)

        let shape_ptr = Pointer[Vector[Int]].alloc(1)
        shape_ptr.store(shape)

        let strides = Vector[Int](_num_dims)
        strides.store(_num_dims - 1, 1)
        for i in range(_num_dims - 1):
            strides.store(
                _num_dims - i - 2,
                strides.load(_num_dims - i - 1) * shape.load(_num_dims - i - 1),
            )
        let strides_ptr = Pointer[Vector[Int]].alloc(1)
        strides_ptr.store(strides)

        let other_params_ptr = Pointer[Vector[Int]].alloc(1)
        other_params_ptr.store(Vector[Int]())

        return Node {
            id_ptr: id_ptr,
            data_id: data_id,
            grad_id: grad_id,
            data: data,
            parents_ptr: parents_ptr,
            children_ptr: children_ptr,
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
            shape_ptr: shape_ptr,
            strides_ptr: strides_ptr,
            other_params_ptr: other_params_ptr,
        }

    fn store_id(self, id: Int):
        self.id_ptr.store(id)

    fn load_id(self) -> Int:
        return self.id_ptr.load()

    fn load_dependencies(self) -> Int:
        return self.dependencies_ptr.load()

    fn incr_dependencies(self):
        self.dependencies_ptr.store(self.dependencies_ptr.load() + 1)

    fn decr_dependencies(self):
        self.dependencies_ptr.store(self.dependencies_ptr.load() - 1)

    fn load_cap(self) -> Int:
        return self.cap_ptr.load()

    fn add_parent(self, node_id: Int):
        let vec = self.parents_ptr.load()
        vec.push_back(node_id)
        self.parents_ptr.store(vec)

    fn add_child(self, node_id: Int):
        let vec = self.children_ptr.load()
        vec.push_back(node_id)
        self.children_ptr.store(vec)

    fn load_parent_id(self, idx: Int) -> Int:
        return self.parents_ptr.load().load(idx)

    fn load_child_id(self, idx: Int) -> Int:
        return self.parents_ptr.load().load(idx)

    fn load_num_parents(self) -> Int:
        return self.parents_ptr.load().len.load()

    fn load_num_children(self) -> Int:
        return self.children_ptr.load().len.load()

    fn load_is_static(self) -> Bool:
        return self.is_static_ptr.load()

    fn load_computed(self) -> Bool:
        return self.computed_ptr.load()

    fn store_computed(self, value: Bool):
        self.computed_ptr.store(value)

    fn is_zero(self) -> Bool:
        for i in range(self.load_cap()):
            if self.load_data(i) != 0.0:
                return False
        return True

    @always_inline
    fn load_data(self, idx: Int) -> Float32:
        return self.data.load().load(idx)

    @always_inline
    fn store_data(self, idx: Int, val: Float32):
        self.data.load().simd_store(idx, val)

    @always_inline
    fn load_data[nelts: Int](self, idx: Int) -> SIMD[DType.float32, nelts]:
        return self.data.load().simd_load[nelts](idx)

    @always_inline
    fn store_data[nelts: Int = 1](self, idx: Int, val: SIMD[DType.float32, nelts]):
        self.data.load().simd_store[nelts](idx, val)

    fn fill(self, val: Float32):
        for i in range(self.load_cap()):
            self.data.load().store(i, val)

    fn fill_incr(self):
        for i in range(self.load_cap()):
            self.data.load(0).store(i, Float32(i))

    @always_inline
    fn load_grad(self, idx: Int) -> Float32:
        return self.data.load(1).load(idx)

    @always_inline
    fn store_grad(self, idx: Int, val: Float32):
        self.data.load(1).simd_store(idx, val)

    @always_inline
    fn load_grad[nelts: Int](self, idx: Int) -> SIMD[DType.float32, nelts]:
        return self.data.load(1).simd_load[nelts](idx)

    @always_inline
    fn store_grad[nelts: Int = 1](self, idx: Int, val: SIMD[DType.float32, nelts]):
        self.data.load(1).simd_store[nelts](idx, val)

    fn grad_fill_incr(self):
        for i in range(self.load_cap()):
            self.data.load(1).store(i, Float32(i))

    fn fill_grad(self, val: Float32):
        for i in range(self.load_cap()):
            self.data.load(1).store(i, val)

    fn initialize(self, data: DTypePointer[DType.float32]):
        self.data.store(0, data)

    fn initialize[
        initialization_function: String, val: Float32 = 0, val2: Float32 = 0
    ](self):
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
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
        let fan_out = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 1)
        let scale = sqrt(2.0 / Float32(fan_in + fan_out))
        self.random_normal(scale, 0.0)

    fn glorot_uniform(self):
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
        let fan_out = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 1)
        let scale = sqrt(6.0 / Float32(fan_in + fan_out))
        self.random_uniform(-scale, scale)

    fn he_normal(self):
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
        let scale = sqrt(2.0 / Float32(fan_in))
        self.random_normal(scale, 0.0)

    fn he_uniform(self):
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
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
            let sigma = sqrt(
                2.0
                / Float32(
                    self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 1)
                )
            )
            self.store_data(i, z * sigma)

    fn identity(self):
        let num_dims = self.num_dims_ptr.load()
        let row: Int = self.shape_ptr.load().load(num_dims - 2)
        let cols: Int = self.shape_ptr.load().load(num_dims - 1)
        let col_strides: Int = (
            self.strides_ptr.load().load(0) * self.shape_ptr.load().load(0)
        ) // cols
        for i in range(col_strides):
            for j in range(cols):
                if i == j:
                    self.store_data(i * cols + j, 1.0)
                else:
                    self.store_data(i * cols + j, 0.0)

    fn lecun_normal(self):
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
        let scale = sqrt(1.0 / Float32(fan_in))
        self.random_normal(scale, 0.0)

    fn lecun_uniform(self):
        let fan_in = self.shape_ptr.load().load(self.shape_ptr.load().len.load() - 2)
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
            self.store_data(i, z * std + mu)

    fn random_uniform(self, min: Float32, max: Float32):
        seed()
        rand(self.data.load(0), self.cap_ptr.load())
        for i in range(self.cap_ptr.load()):
            self.store_data(i, self.load_data(i) * (max - min) + min)

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
                self.store_data(i, z * std + mu)
            else:
                self.store_data(i, 0.0)

    fn zeros(self):
        self.fill(0.0)

    fn orthoganal(self, gain: Float32 = 1.0):
        let num_dims = self.num_dims_ptr.load()
        let row: Int = self.shape_ptr.load().load(num_dims - 2)
        let cols: Int = self.shape_ptr.load().load(num_dims - 1)
        let col_strides: Int = (
            self.strides_ptr.load().load(0) * self.shape_ptr.load().load(0)
        ) // cols
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
                self.store_data(i * cols + j, tmp.load(i * cols + j))

    fn print(self, accuracy: Int = 6):
        let num_dims = self.num_dims_ptr.load()
        let row: Int = self.shape_ptr.load().load(num_dims - 2)
        let cols: Int = self.shape_ptr.load().load(num_dims - 1)
        let col_strides: Int = (
            self.strides_ptr.load().load(0) * self.shape_ptr.load().load(0)
        ) // cols
        print(" ")
        var times = 1
        if self.grad_computed_ptr.load() and self.grad_id.load() != -1:
            times = 2
        for t in range(times):
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
                    for d in range(num_dims - 1):
                        if cols * i % self.strides_ptr.load().load(d) == 0:
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
                                String(self.data.load(t).load(idx))[
                                    :accuracy
                                ] if self.data.load(t).load(idx)
                                != 0.0 else String(0.000)[:accuracy]
                            )
                            if j != cols - 1:
                                print_no_newline(", ")

                    for d in range(num_dims - 2, -1, -1):
                        if cols * (i + 1) % self.strides_ptr.load().load(d) == 0:
                            print_no_newline(" ]")

                    if i < col_strides - 1:
                        print_no_newline(", ")
                        put_new_line()
                    else:
                        print_no_newline(" ], shape: [")
                        for i in range(num_dims):
                            print_no_newline(self.shape_ptr.load().load(i))
                            if i < num_dims - 1:
                                print_no_newline(",")
                        print_no_newline("] ")
                        print_no_newline(">")
        print(" ")
