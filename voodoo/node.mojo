from math import sin, cos, sqrt, log
from random import rand, seed
from .utils import Vector, warn


@register_passable("trivial")
struct Node:
    var id_ptr: Pointer[Int]  # Needs to be a pointer
    var data_id: Pointer[Int]  # Needs to be a pointer
    var grad_id: Pointer[Int]  # Needs to be a pointer
    var data: Pointer[DTypePointer[DType.float32]]  # Needs to be a pointer
    var parents: Vector[Int]
    var children: Vector[Int]
    var dependencies: Int
    var is_static: Bool
    var computed_ptr: Pointer[Bool]  # Needs to be a pointer
    var grad_computed_ptr: Pointer[Bool]  # Needs to be a pointer
    var operator_id: Int
    var grad_operator_id: Int
    var requires_grad: Bool
    var tmp_visited: Bool
    var checkpoint: Bool
    var is_single_ptr: Pointer[Bool]  # Needs to be pointer
    var cap: Int
    var num_dims: Int
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

        let data_id = Pointer[Int].alloc(1)
        data_id.store(-1)

        let grad_id = Pointer[Int].alloc(1)
        grad_id.store(-1)

        let data = Pointer[DTypePointer[DType.float32]].alloc(2)
        data.store(0, DTypePointer[DType.float32].get_null())
        data.store(1, DTypePointer[DType.float32].get_null())

        let parents = Vector[Int]()

        let children = Vector[Int]()

        let dependencies = 0

        let computed_ptr = Pointer[Bool].alloc(1)
        computed_ptr.store(is_static)

        let grad_computed_ptr = Pointer[Bool].alloc(1)
        grad_computed_ptr.store(False)

        let requires_grad = is_static

        let operator_id = -1

        let grad_operator_id = -1

        let tmp_visited = False

        let checkpoint = False

        let is_single_ptr = Pointer[Bool].alloc(1)
        is_single_ptr.store(False)

        let num_dims = shape.len.load()

        var cap = shape.load(0)
        for i in range(1, num_dims):
            cap *= shape.load(i)

        let strides = Vector[Int](num_dims)
        strides.store(num_dims - 1, 1)
        for i in range(num_dims - 1):
            strides.store(
                num_dims - i - 2,
                strides.load(num_dims - i - 1) * shape.load(num_dims - i - 1),
            )

        return Node {
            id_ptr: id_ptr,
            data_id: data_id,
            grad_id: grad_id,
            data: data,
            parents: parents,
            children: children,
            dependencies: dependencies,
            is_static: is_static,
            computed_ptr: computed_ptr,
            grad_computed_ptr: grad_computed_ptr,
            operator_id: operator_id,
            grad_operator_id: grad_operator_id,
            requires_grad: requires_grad,
            tmp_visited: tmp_visited,
            checkpoint: checkpoint,
            is_single_ptr: is_single_ptr,
            cap: cap,
            num_dims: num_dims,
            shape: shape,
            strides: strides,
            other_params: other_params,
        }

    @always_inline
    fn store_id(self, id: Int):
        self.id_ptr.store(id)
    
    @always_inline
    fn load_id(self) -> Int:
        return self.id_ptr.load()
    
    @always_inline
    fn incr_dependencies(inout self):
        self.dependencies += 1
    
    @always_inline
    fn decr_dependencies(inout self):
        self.dependencies -= 1
    
    @always_inline
    fn add_parent(self, node_id: Int):
        self.parents.push_back(node_id)
    
    @always_inline
    fn add_child(self, node_id: Int):
        let vec = self.children.push_back(node_id)
    
    @always_inline
    fn load_parent_id(self, idx: Int) -> Int:
        return self.parents.load(idx)
    
    @always_inline
    fn load_child_id(self, idx: Int) -> Int:
        return self.children.load(idx)
    
    @always_inline
    fn load_num_parents(self) -> Int:
        return self.parents.len.load()
    
    @always_inline
    fn load_num_children(self) -> Int:
        return self.children.len.load()
    
    @always_inline
    fn load_is_static(self) -> Bool:
        return self.is_static
    
    @always_inline
    fn load_computed(self) -> Bool:
        return self.computed_ptr.load()
    
    @always_inline
    fn store_computed(self, value: Bool):
        self.computed_ptr.store(value)
    
    @always_inline
    fn is_zero(self) -> Bool:
        for i in range(self.cap):
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
    fn load_data[NELTS: Int](self, idx: Int) -> SIMD[DType.float32, NELTS]:
        return self.data.load().simd_load[NELTS](idx)

    @always_inline
    fn store_data[NELTS: Int = 1](self, idx: Int, val: SIMD[DType.float32, NELTS]):
        self.data.load().simd_store[NELTS](idx, val)
    
    @always_inline
    fn fill(self, val: Float32):
        for i in range(self.cap):
            self.data.load().store(i, val)
    
    @always_inline
    fn fill_incr(self):
        for i in range(self.cap):
            self.data.load(0).store(i, Float32(i))

    @always_inline
    fn load_grad(self, idx: Int) -> Float32:
        return self.data.load(1).load(idx)

    @always_inline
    fn store_grad(self, idx: Int, val: Float32):
        self.data.load(1).simd_store(idx, val)

    @always_inline
    fn load_grad[NELTS: Int](self, idx: Int) -> SIMD[DType.float32, NELTS]:
        return self.data.load(1).simd_load[NELTS](idx)

    @always_inline
    fn store_grad[NELTS: Int = 1](self, idx: Int, val: SIMD[DType.float32, NELTS]):
        self.data.load(1).simd_store[NELTS](idx, val)
    
    @always_inline
    fn grad_fill_incr(self):
        for i in range(self.cap):
            self.data.load(1).store(i, Float32(i))
    
    @always_inline
    fn fill_grad(self, val: Float32):
        for i in range(self.cap):
            self.data.load(1).store(i, val)
    
    @always_inline
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
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let fan_out = self.shape.load(self.shape.len.load() - 1)
        let scale = sqrt(2.0 / Float32(fan_in + fan_out))
        self.random_normal(scale, 0.0)

    fn glorot_uniform(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let fan_out = self.shape.load(self.shape.len.load() - 1)
        let scale = sqrt(6.0 / Float32(fan_in + fan_out))
        self.random_uniform(-scale, scale)

    fn he_normal(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(2.0 / Float32(fan_in))
        self.random_normal(scale, 0.0)

    fn he_uniform(self):
        let fan_in = self.shape.load(self.shape.len.load() - 2)
        let scale = sqrt(6.0 / Float32(fan_in))
        self.random_uniform(-scale, scale)

    fn he_random(self):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap)
        let u2 = DTypePointer[DType.float32].alloc(self.cap)
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            let sigma = sqrt(2.0 / Float32(self.shape.load(self.shape.len.load() - 1)))
            self.store_data(i, z * sigma)

    fn identity(self):
        let num_dims = self.num_dims
        let row: Int = self.shape.load(num_dims - 2)
        let cols: Int = self.shape.load(num_dims - 1)
        let col_strides: Int = (self.strides.load(0) * self.shape.load(0)) // cols
        for i in range(col_strides):
            for j in range(cols):
                if i == j:
                    self.store_data(i * cols + j, 1.0)
                else:
                    self.store_data(i * cols + j, 0.0)

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
        let u1 = DTypePointer[DType.float32].alloc(self.cap)
        let u2 = DTypePointer[DType.float32].alloc(self.cap)
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            self.store_data(i, z * std + mu)

    fn random_uniform(self, min: Float32, max: Float32):
        seed()
        rand(self.data.load(0), self.cap)
        for i in range(self.cap):
            self.store_data(i, self.load_data(i) * (max - min) + min)

    fn truncated_normal(self, std: Float32 = 1.0, mu: Float32 = 0.0):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self.cap)
        let u2 = DTypePointer[DType.float32].alloc(self.cap)
        rand(u1, self.cap)
        rand(u2, self.cap)
        for i in range(self.cap):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            if z > -2.0 and z < 2.0:
                self.store_data(i, z * std + mu)
            else:
                self.store_data(i, 0.0)

    fn zeros(self):
        self.fill(0.0)

    fn orthoganal(self, gain: Float32 = 1.0):
        let num_dims = self.num_dims
        let row: Int = self.shape.load(num_dims - 2)
        let cols: Int = self.shape.load(num_dims - 1)
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
                self.store_data(i * cols + j, tmp.load(i * cols + j))

    fn print(self, accuracy: Int = 6):
        let num_dims = self.num_dims
        let row: Int = self.shape.load(num_dims - 2)
        let cols: Int = self.shape.load(num_dims - 1)
        let col_strides: Int = (self.strides.load(0) * self.shape.load(0)) // cols
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
                                String(self.data.load(t).load(idx))[
                                    :accuracy
                                ] if self.data.load(t).load(idx)
                                != 0.0 else String(0.000)[:accuracy]
                            )
                            if j != cols - 1:
                                print_no_newline(", ")

                    for d in range(num_dims - 2, -1, -1):
                        if cols * (i + 1) % self.strides.load(d) == 0:
                            print_no_newline(" ]")

                    if i < col_strides - 1:
                        print_no_newline(", ")
                        put_new_line()
                    else:
                        print_no_newline(" ], shape: [")
                        for i in range(num_dims):
                            print_no_newline(self.shape.load(i))
                            if i < num_dims - 1:
                                print_no_newline(",")
                        print_no_newline("] ")
                        print_no_newline(">")
        print(" ")
