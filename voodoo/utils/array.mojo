from memory import memset_zero, memcpy
from math import max


@register_passable("trivial")
struct Vector[type: AnyRegType]:
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

    @always_inline("nodebug")
    fn get_cap(self) -> Int:
        return self._cap

    @always_inline("nodebug")
    fn set_cap(inout self, val: Int):
        self._cap = val

    @always_inline("nodebug")
    fn get_len(self) -> Int:
        return self._len.load()

    @always_inline("nodebug")
    fn set_len(inout self, val: Int):
        self._len.store(val)

    @always_inline("nodebug")
    fn push_back(inout self, elem: type):
        let len = self._len.load()
        let curr_cap = self._cap

        if len == curr_cap:
            self._size_up(curr_cap << 1)

        self._data.store(len, elem)
        self._len.store(len + 1)

    @always_inline("nodebug")
    fn pop_back(inout self) -> type:
        let new_len = self._len.load() - 1
        let curr_cap = self._cap

        self._len.store(new_len)
        let tmp = self._data.load(new_len)

        if new_len <= (curr_cap >> 2) and curr_cap > 32:
            self._size_down(curr_cap >> 1)

        return tmp

    @always_inline("nodebug")
    fn load(self, idx: Int) -> type:
        return self._data.load(idx)

    @always_inline("nodebug")
    fn store(self, idx: Int, value: type):
        self._data.store(idx, value)

    @always_inline("nodebug")
    fn free(owned self):
        self._data.free()
        self._len.free()

    @always_inline("nodebug")
    fn clear(inout self):
        self._size_down(8)
        self._len.store(0)
        self._cap = 8

        memset_zero(self._data, self._cap)

    @always_inline("nodebug")
    fn copy(self) -> Self:
        let len = self._len.load()
        let new_vector = Vector[type](len)

        memcpy(new_vector._data, self._data, len)

        return new_vector

    @always_inline("nodebug")
    fn _size_up(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memset_zero(new_data, new_cap)
        memcpy(new_data, self._data, self._cap)

        self._cap = new_cap
        self._data.free()
        self._data = new_data

    @always_inline("nodebug")
    fn _size_down(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memcpy(new_data, self._data, new_cap)

        self._cap = new_cap
        self._data.free()
        self._data = new_data
