from memory import memset_zero, memcpy

alias nelts = simdwidthof[DType.float32]()


@register_passable("trivial")
struct Vector[type: AnyRegType]:
    var data: Pointer[Pointer[type]]
    var len: Pointer[Int]
    var cap: Pointer[Int]

    fn __init__(_len: Int = 0) -> Self:
        var _cap = _len
        if _len < 8:
            _cap = 8
        let data_ptr = Pointer[type].alloc(_cap)
        memset_zero(data_ptr, _cap)
        let data = Pointer[Pointer[type]].alloc(1)
        data.store(data_ptr)

        let cap = Pointer[Int].alloc(1)
        cap.store(_cap)

        let len = Pointer[Int].alloc(1)
        len.store(_len)

        return Vector[type] {data: data, len: len, cap: cap}

    fn push_back(self, elem: type):
        if self.len.load() == self.cap.load():
            let old_data = self.data.load()
            let new_cap = 2 * self.cap.load()
            let new_data = Pointer[type].alloc(new_cap)
            memset_zero(new_data, new_cap)
            memcpy(new_data, self.data.load(), self.cap.load())
            self.cap.store(new_cap)
            self.data.store(new_data)
            old_data.free()
        self.data.load().store(self.len.load(), elem)
        self.len.store(self.len.load() + 1)

    fn push_back_return(self, elem: type) -> type:
        self.push_back(elem)
        return self

    fn pop_back(self) -> type:
        self.len.store(self.len.load() - 1)
        let tmp = self.data.load().load(self.len.load())

        if self.len.load() <= self.cap.load() // 4 and self.cap.load() > 32:
            let old_data = self.data.load()
            let new_data = Pointer[type].alloc(self.cap.load() // 2)
            memcpy(new_data, self.data.load(), self.cap.load() // 2)
            self.cap.store(self.cap.load() // 2)
            old_data.free()

        return tmp

    fn load(self, idx: Int) -> type:
        return self.data.load().load(idx)

    fn store(self, idx: Int, value: type):
        self.data.load().store(idx, value)

    fn free(self):
        self.data.load().free()
        self.data.free()
        self.len.free()
        self.cap.free()

    fn clear(self):
        self.data.load().free()
        let data = Pointer[type].alloc(8)
        self.data.store(data)
        self.len.store(0)
        self.cap.store(8)

    fn copy(self) -> Self:
        let new_vector = Vector[type](self.len.load())
        memcpy(new_vector.data.load(), self.data.load(), self.len.load())
        return new_vector
