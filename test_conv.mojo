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
    var data = DataLoader("datasets/mnist/mnist.txt")

    let conv1 = Conv2D[in_width=28, in_height=28, in_batches=32, in_channels=1, kernel_size=5, stride=1, padding=0, use_bias=False]()
    let max_pool1 = MaxPool2D[pool_size=2, stride=2, padding=0]()
    let conv2 = Conv2D[in_width=12, in_height=12, in_batches=32, in_channels=1, kernel_size=5, stride=1, padding=0]()
    let max_pool2 = MaxPool2D[pool_size=2, stride=2, padding=0]()
    let flatten = Flatten()
    let dense1 = Dense[in_neurons=16, out_neurons=16, activation="relu"]()
    let dropout = Dropout[dropout_rate=0.1]()
    let dense2 = Dense[in_neurons=16, out_neurons=10, activation="softmax"]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let labels = Tensor(shape(32, 10))
    let images = Tensor(shape(32, 1, 28, 28))

    # INITIALIZE DATA
    
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

    return
    
    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        
        # RESET DATA

        avg_loss += loss.forward_static()[0]
        loss.backward()
        loss.optimize["sgd", 0.01]()

        if epoch % every == 0:
            print(
                "Epoch:",
                epoch,
                " Avg Loss: ",
                avg_loss / every,
                " Time: ",
                nanoseconds_to_seconds(now() - epoch_start),
                "s",
            )
            avg_loss = 0.0

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
