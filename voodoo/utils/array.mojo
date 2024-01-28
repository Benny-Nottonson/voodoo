from memory import memset_zero, memcpy
from math import max
from ..constants import NELTS


@register_passable("trivial")
struct Vector[type: AnyRegType]:
    var data: Pointer[type]
    var len: Pointer[Int]
    var cap: Pointer[Int]

    fn __init__(_len: Int = 0) -> Self:
        let _cap = max(_len, 8)

        let data = Pointer[type].alloc(_cap)
        let cap = Pointer[Int].alloc(1)
        let len = Pointer[Int].alloc(1)
        memset_zero(data, _cap)
        cap.store(_cap)
        len.store(_len)

        return Vector[type] {data: data, len: len, cap: cap}

    @always_inline("nodebug")
    fn size_up(self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)
        memset_zero(new_data, new_cap)
        memcpy(new_data, self.data, self.cap.load())
        self.cap.store(new_cap)

    @always_inline("nodebug")
    fn push_back(self, elem: type):
        if self.len.load() == self.cap.load():
            self.size_up(2 * self.cap.load())
        self.data.store(self.len.load(), elem)
        self.len.store(self.len.load() + 1)

    @always_inline("nodebug")
    fn size_down(self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)
        memcpy(new_data, self.data, new_cap)
        self.cap.store(new_cap)

    @always_inline("nodebug")
    fn pop_back(self) -> type:
        self.len.store(self.len.load() - 1)
        let tmp = self.data.load(self.len.load())

        if self.len.load() <= self.cap.load() // 4 and self.cap.load() > 32:
            self.size_down(self.cap.load() // 2)

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
        self.cap.free()

    @always_inline("nodebug")
    fn clear(self):
        memset_zero(self.data, self.cap.load())
        self.len.store(0)
        self.cap.store(8)

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
