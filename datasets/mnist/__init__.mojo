from builtin.file import open
from voodoo import Tensor, shape
from voodoo.utils import info

alias train_datapath = "./datasets/mnist/MNIST_train.txt"
alias test_datapath = "./datasets/mnist/MNIST_test.txt"
alias nelts = simdwidthof[DType.int8]()

# Data from https://github.com/halimb/MNIST-txt


struct MNist:
    var train_labels: DTypePointer[DType.int8]
    var train_images: Pointer[DTypePointer[DType.int8]]
    var test_labels: DTypePointer[DType.int8]
    var test_images: Pointer[DTypePointer[DType.int8]]

    fn __init__(inout self) raises:
        info("Loading MNIST dataset...\n")

        let train_data = open(train_datapath, "r").read().split("\n")
        let train_size = len(train_data)

        self.train_labels = DTypePointer[DType.int8].alloc(train_size)
        self.train_images = Pointer[DTypePointer[DType.int8]].alloc(train_size)

        for i in range(train_size):
            let line = train_data[i].split(",")
            self.train_labels[i] = atol(line[0])
            self.train_images[i] = DTypePointer[DType.int8].alloc(784)
            for j in range(1, len(line)):
                self.train_images[i][j - 1] = atol(line[j])

        let test_data = open(test_datapath, "r").read().split("\n")
        let test_size = len(test_data)

        self.test_labels = DTypePointer[DType.int8].alloc(test_size)
        self.test_images = Pointer[DTypePointer[DType.int8]].alloc(test_size)

        for i in range(test_size):
            let line = test_data[i].split(",")
            self.test_labels[i] = atol(line[0])
            self.test_images[i] = DTypePointer[DType.int8].alloc(784)
            for j in range(1, len(line)):
                self.test_images[i][j - 1] = atol(line[j])

        info("MNIST dataset loaded.\n")
