from builtin.file import open
from voodoo import Tensor, shape

alias datapath = "./datasets/mnist/mnist.txt"


struct MNist:
    var labels: DynamicVector[Int]
    var images: DynamicVector[DynamicVector[Int]]

    fn __init__(inout self) raises:
        self.labels = DynamicVector[Int]()
        self.images = DynamicVector[DynamicVector[Int]]()

        let file = open(datapath, "r").read().split("\n")

        for i in range(len(file)):
            let line = file[i].split(" ")
            self.labels.push_back(atol(line[0]))
            self.images.push_back(DynamicVector[Int]())
            for j in range(1, len(line)):
                self.images[i].push_back(atol(line[j]))
