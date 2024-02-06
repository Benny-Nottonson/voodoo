from memory import memset_zero, memcpy
from tensor import TensorShape
from math import max


@register_passable("trivial")
struct Vector[type: AnyRegType](Sized):
    var _data: Pointer[type]
    var _len: Pointer[Int]
    var _cap: Int

    fn __init__(len: Int = 0) -> Self:
        let _cap = max(len, 8)
        let _data = Pointer[type].alloc(_cap)
        let _len = Pointer[Int].alloc(1)

        memset_zero(_data, _cap)
        _len.store(len)

        return Vector[type] {_data: _data, _len: _len, _cap: _cap}

    fn __init__(shape: TensorShape) -> Self:
        let len = shape.rank()
        let _data = Pointer[type].alloc(len)
        let _len = Pointer[Int].alloc(1)

        for i in range(len):
            _data.store(i, shape[i])

        _len.store(len)

        return Vector[type] {_data: _data, _len: _len, _cap: len}

    fn __len__(self) -> Int:
        return self._len.load()

    fn __getitem__(self, idx: Int) -> type:
        return self._data.load(idx)

    fn __setitem__(self, idx: Int, value: type):
        self._data.store(idx, value)

    fn push_back(inout self, elem: type):
        let len = self._len.load()
        let curr_cap = self._cap

        if len == curr_cap:
            self._resize[True](max(1, curr_cap << 1))

        self._data.store(len, elem)
        self._len.store(len + 1)

    fn pop_back(inout self) -> type:
        let new_len = self._len.load() - 1
        let curr_cap = self._cap

        self._len.store(new_len)
        let tmp = self._data.load(new_len)

        if new_len <= (curr_cap >> 2) and curr_cap > 32:
            self._resize[False](curr_cap >> 1)

        return tmp

    fn free(owned self):
        self._data.free()
        self._len.free()

    fn clear(inout self):
        self._resize[False](8)
        self._len.store(0)

        memset_zero(self._data, self._cap)

    fn copy(self) -> Self:
        let len = self._len.load()
        let new_vector = Vector[type](len)

        memcpy(new_vector._data, self._data, len)

        return new_vector

    fn _resize[up: Bool](inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        @parameter
        if up:
            memset_zero(new_data, new_cap)
            memcpy(new_data, self._data, self._cap)
        else:
            memcpy(new_data, self._data, new_cap)

        self._cap = new_cap
        self._data.free()
        self._data = new_data


fn reduce_vector_mul[v: Vector[Int]]() -> Int:
    var result = 1

    for i in range(len(v)):
        result *= v[i]

    return result
