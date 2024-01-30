from memory import memset_zero, memcpy
from math import max
from ..constants import NELTS


@register_passable("trivial")
struct Vector[type: AnyRegType]:
    var data: Pointer[type]
    var len: Pointer[Int]
    var internal_cap: Pointer[Int]

    fn __init__(_len: Int = 0) -> Self:
        let _cap = max(_len, 8)

        let data = Pointer[type].alloc(_cap)
        let cap = Pointer[Int].alloc(1)
        let len = Pointer[Int].alloc(1)
        memset_zero(data, _cap)
        cap.store(_cap)
        len.store(_len)

        return Vector[type] {data: data, len: len, internal_cap: cap}

    @always_inline("nodebug")
    fn get_cap(self) -> Int:
        return self.internal_cap.load()

    @always_inline("nodebug")
    fn set_cap(self, val: Int):
        self.internal_cap.store(val)

    @always_inline("nodebug")
    fn size_up(self, new_cap: Int):
        # TODO: THIS IS THE PROBLEM, NOT ACTUALLY CHANGING THE DATA
        let new_data = Pointer[type].alloc(new_cap)
        memset_zero(new_data, new_cap)
        memcpy(new_data, self.data, self.get_cap())
        self.set_cap(new_cap)

    @always_inline("nodebug")
    fn push_back(self, elem: type):
        if self.len.load() == self.get_cap():
            self.size_up(2 * self.get_cap())
        self.data.store(self.len.load(), elem)
        self.len.store(self.len.load() + 1)

    @always_inline("nodebug")
    fn size_down(self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)
        memcpy(new_data, self.data, new_cap)
        self.set_cap(new_cap)

    @always_inline("nodebug")
    fn pop_back(self) -> type:
        self.len.store(self.len.load() - 1)
        let tmp = self.data.load(self.len.load())

        if self.len.load() <= self.get_cap() // 4 and self.get_cap() > 32:
            self.size_down(self.get_cap() // 2)

        return tmp

    @always_inline("nodebug")
    fn load(self, idx: Int) -> type:
        return self.data.load(idx)

    @always_inline("nodebug")
    fn store(self, idx: Int, value: type):
        self.data.store(idx, value)

    @always_inline("nodebug")
    fn free(self):
        self.data.free()
        self.len.free()
        self.internal_cap.free()

    @always_inline("nodebug")
    fn clear(self):
        self.size_down(8)
        self.len.store(0)
        self.set_cap(8)
        memset_zero(self.data, self.get_cap())

    @always_inline("nodebug")
    fn copy(self) -> Self:
        let len = self.len.load()
        let new_vector = Vector[type](len)
        memcpy(new_vector.data, self.data, len)
        return new_vector

    @always_inline("nodebug")
    fn get_transposed(self) -> Self:
        let new_shape = self.copy()
        let len = self.len.load()
        new_shape.store(len - 2, self.load(len - 1))
        new_shape.store(len - 1, self.load(len - 2))
        return new_shape
