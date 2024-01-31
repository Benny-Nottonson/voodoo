from math import sin, cos, sqrt, log, iota
from random import rand, seed
from .utils import Vector, warn


@register_passable("trivial")
struct Node:
    var _id_ptr: Pointer[Int]
    var _data_id_ptr: Pointer[Int]
    var _grad_id_ptr: Pointer[Int]
    var _data_ptr: Pointer[DTypePointer[DType.float32]]
    var _parents: Vector[Int]
    var _children: Vector[Int]
    var _dependencies_ptr: Pointer[Int]
    var _is_static_ptr: Pointer[Bool]
    var _computed_ptr: Pointer[Bool]
    var _grad_computed_ptr: Pointer[Bool]
    var _operator_id_ptr: Pointer[Int]
    var _grad_operator_id_ptr: Pointer[Int]
    var _tmp_visited_ptr: Pointer[Bool]
    var _checkpoint_ptr: Pointer[Bool]
    var _is_single_ptr: Pointer[Bool]
    var _cap_ptr: Pointer[Int]
    var _num_dims_ptr: Pointer[Int]
    var _shape: Vector[Int]
    var _strides: Vector[Int]
    var _other_params: Vector[Int]

    fn __init__(
        id: Int,
        shape: Vector[Int],
        is_static: Bool = True,
        other_params: Vector[Int] = Vector[Int](),
    ) raises -> Self:
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
        num_dims_ptr.store(len(shape))
        let cap_ptr = Pointer[Int].alloc(1)
        cap_ptr.store(1)

        for i in range(len(shape)):
            cap_ptr.store(cap_ptr.load() * shape.load(i))

        let strides = Vector[Int](len(shape))
        strides.store(len(shape) - 1, 1)
        for i in range(len(shape) - 1):
            strides.store(
                len(shape) - i - 2,
                strides.load(len(shape) - i - 1) * shape.load(len(shape) - i - 1),
            )

        return Node {
            _id_ptr: id_ptr,
            _data_id_ptr: data_id_ptr,
            _grad_id_ptr: grad_id_ptr,
            _data_ptr: data_ptr,
            _parents: parents,
            _children: children,
            _dependencies_ptr: dependencies_ptr,
            _is_static_ptr: is_static_ptr,
            _computed_ptr: computed_ptr,
            _grad_computed_ptr: grad_computed_ptr,
            _operator_id_ptr: operator_id_ptr,
            _grad_operator_id_ptr: grad_operator_id_ptr,
            _tmp_visited_ptr: tmp_visited_ptr,
            _checkpoint_ptr: checkpoint_ptr,
            _is_single_ptr: is_single_ptr,
            _cap_ptr: cap_ptr,
            _num_dims_ptr: num_dims_ptr,
            _shape: shape,
            _strides: strides,
            _other_params: other_params,
        }

    @always_inline("nodebug")
    fn get_id(self) -> Int:
        return self._id_ptr.load()

    @always_inline("nodebug")
    fn set_id(self, id: Int):
        self._id_ptr.store(id)

    @always_inline("nodebug")
    fn get_data_id(self) -> Int:
        return self._data_id_ptr.load()

    @always_inline("nodebug")
    fn set_data_id(self, id: Int):
        self._data_id_ptr.store(id)

    @always_inline("nodebug")
    fn get_grad_id(self) -> Int:
        return self._grad_id_ptr.load()

    @always_inline("nodebug")
    fn set_grad_id(self, id: Int):
        self._grad_id_ptr.store(id)

    @always_inline("nodebug")
    fn get_data(self) -> DTypePointer[DType.float32]:
        return self._data_ptr.load(0)

    @always_inline("nodebug")
    fn set_data(self, data: DTypePointer[DType.float32]):
        self._data_ptr.store(0, data)

    @always_inline("nodebug")
    fn get_grad(self) -> DTypePointer[DType.float32]:
        return self._data_ptr.load(1)

    @always_inline("nodebug")
    fn set_grad(self, grad: DTypePointer[DType.float32]):
        self._data_ptr.store(1, grad)

    @always_inline("nodebug")
    fn get_parents(self) -> Vector[Int]:
        return self._parents

    @always_inline("nodebug")
    fn push_back_parent(inout self, parent: Int):
        self._parents.push_back(parent)

    @always_inline("nodebug")
    fn clear_parents(inout self):
        self._parents.clear()

    @always_inline("nodebug")
    fn get_children(self) -> Vector[Int]:
        return self._children

    @always_inline("nodebug")
    fn push_back_child(inout self, child: Int):
        self._children.push_back(child)

    @always_inline("nodebug")
    fn clear_children(inout self):
        self._children.clear()

    @always_inline("nodebug")
    fn get_dependencies(self) -> Int:
        return self._dependencies_ptr.load()

    @always_inline("nodebug")
    fn set_dependencies(self, dependencies: Int):
        self._dependencies_ptr.store(dependencies)

    @always_inline("nodebug")
    fn get_is_static(self) -> Bool:
        return self._is_static_ptr.load()

    @always_inline("nodebug")
    fn set_is_static(self, is_static: Bool):
        self._is_static_ptr.store(is_static)

    @always_inline("nodebug")
    fn get_computed(self) -> Bool:
        return self._computed_ptr.load()

    @always_inline("nodebug")
    fn set_computed(self, computed: Bool):
        self._computed_ptr.store(computed)

    @always_inline("nodebug")
    fn get_grad_computed(self) -> Bool:
        return self._grad_computed_ptr.load()

    @always_inline("nodebug")
    fn set_grad_computed(self, grad_computed: Bool):
        self._grad_computed_ptr.store(grad_computed)

    @always_inline("nodebug")
    fn get_operator_id(self) -> Int:
        return self._operator_id_ptr.load()

    @always_inline("nodebug")
    fn set_operator_id(self, operator_id: Int):
        self._operator_id_ptr.store(operator_id)

    @always_inline("nodebug")
    fn get_grad_operator_id(self) -> Int:
        return self._grad_operator_id_ptr.load()

    @always_inline("nodebug")
    fn set_grad_operator_id(self, grad_operator_id: Int):
        self._grad_operator_id_ptr.store(grad_operator_id)

    @always_inline("nodebug")
    fn get_tmp_visited(self) -> Bool:
        return self._tmp_visited_ptr.load()

    @always_inline("nodebug")
    fn set_tmp_visited(self, tmp_visited: Bool):
        self._tmp_visited_ptr.store(tmp_visited)

    @always_inline("nodebug")
    fn get_checkpoint(self) -> Bool:
        return self._checkpoint_ptr.load()

    @always_inline("nodebug")
    fn set_checkpoint(self, checkpoint: Bool):
        self._checkpoint_ptr.store(checkpoint)

    @always_inline("nodebug")
    fn get_is_single(self) -> Bool:
        return self._is_single_ptr.load()

    @always_inline("nodebug")
    fn set_is_single(self, is_single: Bool):
        self._is_single_ptr.store(is_single)

    @always_inline("nodebug")
    fn get_cap(self) -> Int:
        return self._cap_ptr.load()

    @always_inline("nodebug")
    fn get_num_dims(self) -> Int:
        return self._num_dims_ptr.load()

    @always_inline("nodebug")
    fn get_shape(self) -> Vector[Int]:
        return self._shape

    @always_inline("nodebug")
    fn get_strides(self) -> Vector[Int]:
        return self._strides

    @always_inline("nodebug")
    fn get_other_params(self) -> Vector[Int]:
        return self._other_params

    @always_inline("nodebug")
    fn is_zero(self) -> Bool:
        for i in range(self._cap_ptr.load()):
            if self._data_ptr.load(0).load(i) != 0.0:
                return False
        return True

    @always_inline("nodebug")
    fn fill(self, val: Float32):
        for i in range(self._cap_ptr.load()):
            self._data_ptr.load(0).store(i, val)

    # Use math.iota here https://github.com/rd4com/mojo-learning/blob/main/tutorials/simd.md
    @always_inline("nodebug")
    fn fill_incr(self):
        iota(self._data_ptr.load(0), self._cap_ptr.load())

    @always_inline("nodebug")
    fn fill_grad(self, val: Float32):
        for i in range(self._cap_ptr.load()):
            self._data_ptr.load(1).store(i, val)

    @always_inline("nodebug")
    fn grad_fill_incr(self):
        iota(self._data_ptr.load(1), self._cap_ptr.load())

    fn initialize[
        initialization_function: String, val: Float32 = 0, val2: Float32 = 0
    ](self) raises:
        @parameter
        if initialization_function == "he_normal":
            self.he_normal()
        elif initialization_function == "random_uniform":
            self.random_uniform(val, val2)
        elif initialization_function == "random_normal":
            self.random_normal(val, val2)
        elif initialization_function == "ones":
            self.fill(1.0)
        elif initialization_function == "zeros":
            self.fill(0.0)
        else:
            warn(
                "Invalid initialization function: "
                + initialization_function
                + " using zeros\n"
            )
            self.fill(0.0)

    fn he_normal(self) raises:
        let fan_in: Float32 = self._shape.load(len(self._shape) - 2)
        let scale = sqrt(2.0 / fan_in)
        self.random_normal(scale, 0.0)

    fn random_normal(self, std: Float32 = 1.0, mu: Float32 = 0.0):
        seed()
        let pi = 3.14159265358979
        let u1 = DTypePointer[DType.float32].alloc(self._cap_ptr.load())
        let u2 = DTypePointer[DType.float32].alloc(self._cap_ptr.load())
        rand(u1, self._cap_ptr.load())
        rand(u2, self._cap_ptr.load())
        for i in range(self._cap_ptr.load()):
            let z = sqrt(-2.0 * log(u1.load(i))) * cos(2.0 * pi * u2.load(i))
            self._data_ptr.load(0).store(i, z * std + mu)

    fn random_uniform(self, min: Float32, max: Float32):
        seed()
        rand(self._data_ptr.load(0), self._cap_ptr.load())
        for i in range(self._cap_ptr.load()):
            self._data_ptr.load(0).store(
                i, self._data_ptr.load(0).load(i) * (max - min) + min
            )

    fn free(self):
        self._id_ptr.free()
        self._data_id_ptr.free()
        self._grad_id_ptr.free()
        self._data_ptr.load(0).free()
        self._data_ptr.load(1).free()
        self._data_ptr.free()
        self._parents.free()
        self._children.free()
        self._dependencies_ptr.free()
        self._is_static_ptr.free()
        self._computed_ptr.free()
        self._grad_computed_ptr.free()
        self._operator_id_ptr.free()
        self._grad_operator_id_ptr.free()
        self._tmp_visited_ptr.free()
        self._checkpoint_ptr.free()
        self._is_single_ptr.free()
        self._cap_ptr.free()
        self._num_dims_ptr.free()
        self._shape.free()
        self._strides.free()
        self._other_params.free()

    fn print(self, accuracy: Int = 6) raises:
        let row: Int = self._shape.load(self._num_dims_ptr.load() - 2)
        let cols: Int = self._shape.load(self._num_dims_ptr.load() - 1)
        let col_strides: Int = (self._strides.load(0) * self._shape.load(0)) // cols
        print(" ")
        var times = 1
        if self._grad_computed_ptr.load() and self._grad_id_ptr.load() != -1:
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
                for d in range(self._num_dims_ptr.load() - 1):
                    if cols * i % self._strides.load(d) == 0:
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
                            String(self._data_ptr.load(0).load(idx))[
                                :accuracy
                            ] if self._data_ptr.load(0).load(idx)
                            != 0.0 else String(0.000)[:accuracy]
                        )
                        if j != cols - 1:
                            print_no_newline(", ")

                for d in range(self._num_dims_ptr.load() - 2, -1, -1):
                    if cols * (i + 1) % self._strides.load(d) == 0:
                        print_no_newline(" ]")

                if i < col_strides - 1:
                    print_no_newline(", ")
                    put_new_line()
                else:
                    print_no_newline(" ], shape: [")
                    for i in range(self._num_dims_ptr.load()):
                        print_no_newline(self._shape.load(i))
                        if i < self._num_dims_ptr.load() - 1:
                            print_no_newline(",")
                    print_no_newline("] ")
                    print_no_newline(">")
        print(" ")
