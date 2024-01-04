from voodoo import Tensor, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from voodoo.layers.Conv2D import Conv2D
from voodoo.layers.MaxPool2D import MaxPool2D
from voodoo.layers.Flatten import Flatten
from voodoo.layers.Dense import Dense
from voodoo.layers.Dropout import Dropout
from time.time import now
from datasets import DataLoader


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    let d = DataLoader("datasets/mnist/mnist.txt")

    let conv1 = Conv2D[in_channels=1, out_channels=1, kernel_width=5, kernel_height=5, stride=1, padding=0, use_bias=False]()
    let max_pool1 = MaxPool2D[pool_size=2, stride=2, padding=0]()
    let conv2 = Conv2D[in_channels=1, out_channels=1, kernel_width=5, kernel_height=5, stride=1, padding=0, use_bias=False]()
    let max_pool2 = MaxPool2D[pool_size=2, stride=2, padding=0]()
    let flatten = Flatten()
    let dense1 = Dense[in_neurons=16, out_neurons=16, activation="relu"]()
    let dropout = Dropout[dropout_rate=0.1]()
    let dense2 = Dense[in_neurons=16, out_neurons=10, activation="softmax"]()

    var avg_loss: Float32 = 0.0
    let every = 1
    let num_epochs = 10

    let labels = Tensor(shape(32, 10))
    let images = Tensor(shape(32, 1, 28, 28))

    for image in range(0, 32):
        for pixel in range(785):
            let data = d.data.load(image * 785 + pixel)
            if pixel == 0:
                labels.store((image * 10 + data).to_int(), 1.0)
            else:
                images.store(image * 784 + pixel - 1, data / 255.0)

    var x = conv1.forward(images)
    x = max_pool1.forward(x)
    x = conv2.forward(x)
    x = max_pool2.forward(x)
    x = flatten.forward(x)
    x = dense1.forward(x)
    x = dropout.forward(x)
    x = dense2.forward(x)
    let loss = x.compute_loss[get_loss_code["cce"]()](labels)
    
    loss.print()
    loss.backward()
    loss.optimize["sgd", 0.01]()
    loss.print()