from memory import memset_zero, memcpy
from math import max
from sys.ffi import external_call
from ..constants import NELTS


@register_passable("trivial")
struct Vector[type: AnyRegType]:
    var data: Pointer[type]
    var internal_len: Pointer[Int]
    var internal_cap: Int

    fn __init__(len: Int = 0) -> Self:
        let internal_cap = max(len, 8)
        let data = Pointer[type].alloc(internal_cap)
        let internal_len = Pointer[Int].alloc(1)
        
        memset_zero(data, internal_cap)
        internal_len.store(len)

        return Vector[type] {data: data, internal_len: internal_len, internal_cap: internal_cap}

    @always_inline("nodebug")
    fn get_cap(self) -> Int:
        return self.internal_cap

    @always_inline("nodebug")
    fn set_cap(inout self, val: Int):
        self.internal_cap = val

    @always_inline("nodebug")
    fn get_len(self) -> Int:
        return self.internal_len.load()

    @always_inline("nodebug")
    fn set_len(inout self, val: Int):
        self.internal_len.store(val)

    @always_inline("nodebug")
    fn size_up(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memset_zero(new_data, new_cap)
        memcpy(new_data, self.data, self.get_cap())

        self.set_cap(new_cap)
        self.data.free()
        self.data = new_data

    @always_inline("nodebug")
    fn push_back(inout self, elem: type):
        let len = self.get_len()
        let curr_cap = self.get_cap()
        
        if len == curr_cap:
            self.size_up(curr_cap << 1)

        self.data.store(len, elem)
        self.set_len(len + 1)

    @always_inline("nodebug")
    fn size_down(inout self, new_cap: Int):
        let new_data = Pointer[type].alloc(new_cap)

        memcpy(new_data, self.data, new_cap)

        self.set_cap(new_cap)
        self.data.free()
        self.data = new_data

    @always_inline("nodebug")
    fn pop_back(inout self) -> type:
        let new_len = self.get_len() - 1
        let curr_cap = self.get_cap()

        self.set_len(new_len)
        let tmp = self.data.load(new_len)

        if new_len <= (curr_cap >> 2) and curr_cap > 32:
            self.size_down(curr_cap >> 1)

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
        self.internal_len.free()

    @always_inline("nodebug")
    fn clear(inout self):
        self.size_down(8)
        self.set_len(0)
        self.set_cap(8)

        memset_zero(self.data, self.get_cap())

    @always_inline("nodebug")
    fn copy(self) -> Self:
        let len = self.get_len()
        let new_vector = Vector[type](len)

        memcpy(new_vector.data, self.data, len)

        return new_vector
