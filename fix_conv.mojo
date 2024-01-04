from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now
from datasets import DataLoader


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    let conv1 = Layer[type="conv2d", in_width=28, in_height=28, in_batches=32, in_channels=1, kernel_width=5, kernel_height=5, stride=1, padding=0, use_bias=False]()
    let max_pool1 = Layer[type="maxpool2d", pool_size=2, stride=2, padding=0]()
    let conv2 = Layer[type="conv2d", in_width=12, in_height=12, in_batches=32, in_channels=1, kernel_width=5, kernel_height=5, stride=1, padding=0]()
    let max_pool2 = Layer[type="maxpool2d", pool_size=2, stride=2, padding=0]()
    let flatten = Layer[type="flatten"]()
    let dense1 = Layer[type="dense", in_neurons=16, out_neurons=16, activation="relu"]()
    let dropout = Layer[type="dropout", dropout_rate=0.1]()
    let dense2 = Layer[type="dense", in_neurons=16, out_neurons=10, activation="softmax"]()

    var mnist_data = DataLoader("datasets/mnist/mnist.txt")

    let images_tensor = Tensor(shape(32, 1, 28, 28))
    let label_tensor = Tensor(shape(32, 1, 10)).initialize["zeros"]()

    for i in range(32):
        let label = mnist_data.load(32, 1, 1)
        let image = mnist_data.load(32, 1, 785)
        label_tensor.store((i * 32 + label.load()).to_int(), 1)
        for i in range(784):
            images_tensor.store(i, image.load(i))

    label_tensor.print()
    images_tensor.print()
    return
    
    var x = conv1.forward(images_tensor)
    x = max_pool1.forward(x)
    x = conv2.forward(x)
    x = max_pool2.forward(x)
    x = flatten.forward(x)
    x = dense1.forward(x)
    x = dropout.forward(x)
    x = dense2.forward(x)
    x.print()