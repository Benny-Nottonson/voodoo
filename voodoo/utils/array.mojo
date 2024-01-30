from memory import memset_zero, memcpy
from math import max


@register_passable("trivial")
struct Vector[type: AnyRegType]:
    """
    A memory efficient implementation of a dynamically sized vector, passable to registers.
    """

    var _data: Pointer[type]
    var _len: Pointer[Int]
    var _cap: Int

    fn __init__(len: Int = 0) -> Self:
        let _cap = max(len, 8)
        let data = Pointer[type].alloc(_cap)
        let _len = Pointer[Int].alloc(1)

        memset_zero(data, _cap)
        _len.store(len)

        return Vector[type] {_data: data, _len: _len, _cap: _cap}

    @always_inline("nodebug")
    fn get_data(self) -> Pointer[type]:
        return self._data

    @always_inline("nodebug")
    fn set_data(inout self, val: Pointer[type]):
        self._data = val

    @always_inline("nodebug")
    fn free_data(self):
        self._data.free()

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
    fn free_len(self):
        self._len.free()

    @always_inline("nodebug")
    fn push_back(inout self, elem: type):
        let len = self.get_len()
        let curr_cap = self.get_cap()

        if len == curr_cap:
            self._size_up(curr_cap << 1)

        self.get_data().store(len, elem)
        self.set_len(len + 1)

    @always_inline("nodebug")
    fn pop_back(inout self) -> type:
        let new_len = self.get_len() - 1
        let curr_cap = self.get_cap()

        self.set_len(new_len)
        let tmp = self.get_data().load(new_len)

        if new_len <= (curr_cap >> 2) and curr_cap > 32:
            self._size_down(curr_cap >> 1)

        return tmp

    @always_inline("nodebug")
    fn _size_up(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memset_zero(new_data, new_cap)
        memcpy(new_data, self.get_data(), self.get_cap())

        self.set_cap(new_cap)
        self.free_data()
        self.set_data(new_data)

    @always_inline("nodebug")
    fn _size_down(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memcpy(new_data, self.get_data(), new_cap)

        self.set_cap(new_cap)
        self.free_data()
        self.set_data(new_data)

    @always_inline("nodebug")
    fn load(self, idx: Int) -> type:
        return self.get_data().load(idx)

    @always_inline("nodebug")
    fn store(self, idx: Int, value: type):
        self.get_data().store(idx, value)

    @always_inline("nodebug")
    fn free(owned self):
        self.free_data()
        self.free_len()

    @always_inline("nodebug")
    fn clear(inout self):
        self._size_down(8)
        self.set_len(0)
        self.set_cap(8)

        memset_zero(self.get_data(), self.get_cap())

    @always_inline("nodebug")
    fn copy(self) -> Self:
        let len = self.get_len()
        let new_vector = Vector[type](len)

        memcpy(new_vector.get_data(), self.get_data(), len)

        return new_vector
