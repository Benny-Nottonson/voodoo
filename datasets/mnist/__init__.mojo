from voodoo.utils import info

alias train_datapath = "./datasets/mnist/MNIST_train.csv"
alias test_datapath = "./datasets/mnist/MNIST_test.csv"
alias NELTS = simdwidthof[DType.int8]()

# Data from https://github.com/halimb/MNIST-txt


struct MNist:
    var train_labels: DTypePointer[DType.int8]
    var train_images: Pointer[DTypePointer[DType.int8]]
    var test_labels: DTypePointer[DType.int8]
    var test_images: Pointer[DTypePointer[DType.int8]]

    fn __init__(inout self) raises:
        info("Loading MNIST dataset...\n")

        var train_data = open(train_datapath, "r").read().split("\n")
        var train_size = len(train_data) - 1

        self.train_labels = DTypePointer[DType.int8].alloc(train_size)
        self.train_images = Pointer[DTypePointer[DType.int8]].alloc(train_size)

        for i in range(train_size):
            var line = train_data[i].strip().split(",")
            self.train_labels[i] = atol(line[0])
            self.train_images[i] = DTypePointer[DType.int8].alloc(784)
            for j in range(1, len(line)):
                self.train_images[i][j - 1] = atol(line[j])

        var test_data = open(test_datapath, "r").read().split("\n")
        var test_size = len(test_data) - 1

        self.test_labels = DTypePointer[DType.int8].alloc(test_size)
        self.test_images = Pointer[DTypePointer[DType.int8]].alloc(test_size)

        for i in range(test_size):
            var line = test_data[i].strip().split(",")
            self.test_labels[i] = atol(line[0])
            self.test_images[i] = DTypePointer[DType.int8].alloc(784)
            for j in range(1, len(line)):
                self.test_images[i][j - 1] = atol(line[j])

        info("MNIST dataset loaded.\n")
        print("There are ", train_size, " training samples and ", test_size, " test samples.\n")
