from random import seed, randint

fn is_int(s: String) -> Bool:
    if (
        s == "0"
        or s == "1"
        or s == "2"
        or s == "3"
        or s == "4"
        or s == "5"
        or s == "6"
        or s == "7"
        or s == "8"
        or s == "9"
    ):
        return True
    return False

struct DataLoader:
    var indeces: DTypePointer[DType.int32]
    var data: DTypePointer[DType.float32]
    var file_path: String
    var rows: Int
    var cols: Int
    var counter: Int

    fn __init__(inout self, file_path: String) raises:
        self.file_path = file_path

        var vec = DynamicVector[Float32]()
        var rows: Int = 1
        var cols: Int = 0
        with open(file_path, "r") as f:
            let text: String = f.read()
            var connected: String = ""
            for s in range(len(text)):
                let c: String = text[s]
                if is_int(c):
                    connected += c

                if not is_int(c) or c == "\n" or c == " ":
                    vec.push_back(Float32(atol(connected)))
                    connected = ""
                    if rows == 1:
                        cols += 1
                if c == "\n" or c == "$":
                    rows += 1

        let size = len(vec)

        self.data = DTypePointer[DType.float32].alloc(size)

        for i in range(size):
            self.data.store(i, vec[i])
        vec.clear()

        self.rows = rows
        self.cols = cols
        self.indeces = DTypePointer[DType.int32].alloc(self.rows)
        self.counter = 0
        seed()
        randint[DType.int32](self.indeces, self.rows, 0, self.rows - 1)

    fn load(
        inout self,
        batch_size: Int,
        start: Int,
        end: Int,
        scalingFactor: Float32 = Float32(1.0),
    ) raises -> DTypePointer[DType.float32]:
        var _start = start
        var _end = end
        let _batch_size = batch_size
        if _start < 0:
            _start = 0
        if _end > self.cols:
            _end = self.cols

        let batch = DTypePointer[DType.float32].alloc(_batch_size * (_end - _start))
        if _batch_size < self.rows and _batch_size * (self.counter + 1) < self.rows:
            self.counter += 1
            for i in range(_batch_size):
                let sampleIndex = self.indeces.load(
                    (self.counter - 1) * _batch_size + i
                ).to_int()
                for j in range(_start, _end):
                    batch.store(
                        i * (_end - _start) + j - _start,
                        scalingFactor * self.data.load(sampleIndex * self.cols + j),
                    )
        elif _batch_size < self.rows:
            seed()
            randint[DType.int32](self.indeces, self.rows, 0, self.rows - 1)
            self.counter = 1
            for i in range(_batch_size):
                let sampleIndex = self.indeces.load(
                    (self.counter - 1) * _batch_size + i
                ).to_int()
                for j in range(_start, _end):
                    batch.store(
                        i * (_end - _start) + j - _start,
                        scalingFactor * self.data.load(sampleIndex * self.cols + j),
                    )
        else:
            print("Error: batch_size exceeds the number of samples in the data set!")

        return batch

    fn load_again(
        inout self,
        batch_size: Int,
        start: Int,
        end: Int,
        scalingFactor: Float32 = Float32(1.0),
    ) raises -> DTypePointer[DType.float32]:
        var _start = start
        var _end = end
        let _batch_size = batch_size
        if _start < 0:
            _start = 0
        if _end > self.cols:
            _end = self.cols

        let batch = DTypePointer[DType.float32].alloc(_batch_size * (_end - _start))
        if _batch_size < self.rows and _batch_size * (self.counter) < self.rows:
            for i in range(_batch_size):
                let sampleIndex = self.indeces.load(
                    (self.counter - 1) * _batch_size + i
                ).to_int()
                for j in range(_start, _end):
                    batch.store(
                        i * (_end - _start) + j - _start,
                        scalingFactor * self.data.load(sampleIndex * self.cols + j),
                    )

        return batch

    fn one_hot(
        inout self, batch_size: Int, index: Int, ndims: Int
    ) raises -> DTypePointer[DType.float32]:
        let _batch_size = batch_size
        let batch = DTypePointer[DType.float32].alloc(_batch_size * ndims)

        for i in range(_batch_size):
            let sampleIndex = self.indeces.load(
                (self.counter - 1) * _batch_size + i
            ).to_int()
            let entry = self.data.load(sampleIndex * self.cols + index).to_int()
            for j in range(ndims):
                if entry == j:
                    batch.store(i * ndims + j, 1)
                else:
                    batch.store(i * ndims + j, 0)

        return batch