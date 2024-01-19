from voodoo import Tensor, get_loss_code, Graph
from voodoo.utils.shape import shape
from voodoo.layers.Dense import Dense
from voodoo.layers.Conv2D import Conv2D
from voodoo.layers.Reshape import Reshape
from voodoo.layers.MaxPool2D import MaxPool2D
from voodoo.layers.Dropout import Dropout
from voodoo.utils import (
    info,
    clear,
)
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


alias data_shape = shape(16, 1, 28, 28)


fn main() raises:
    let conv_layer_one = Conv2D[
        in_channels=1,
        kernel_width=5,
        kernel_height=5,
        stride=1,
        padding=0,
        bias_initializer="he_normal",
    ]()
    let max_pool_one = MaxPool2D[
        kernel_width=2,
        kernel_height=2,
        stride=2,
    ]()
    let conv_layer_two = Conv2D[
        in_channels=1,
        kernel_width=5,
        kernel_height=5,
        stride=1,
        padding=0,
        bias_initializer="he_normal",
    ]()
    let max_pool_two = MaxPool2D[
        kernel_width=2,
        kernel_height=2,
        stride=2,
    ]()
    let flatten = Reshape[shape(32, 16)]()
    let dense_one = Dense[
        in_neurons=16,
        out_neurons=16,
        activation="relu",
        bias_initializer="he_normal",
    ]()
    let dropout = Dropout[dropout_rate=0.1,]()
    let dense_two = Dense[
        in_neurons=16,
        out_neurons=10,
        activation="relu",
        bias_initializer="he_normal",
    ]()

    var avg_loss: Float32 = 0.0
    let every = 100
    let num_epochs = 2000

    let input = Tensor(data_shape).initialize["he_normal", 0, 1]().dynamic()
    let true_vals = Tensor(shape(32, 10))
    var x = conv_layer_one.forward(input)
    x = max_pool_one.forward(x)
    x = conv_layer_two.forward(x)
    x = max_pool_two.forward(x)
    x = flatten.forward(x)
    x = dense_one.forward(x)
    x = dropout.forward(x)
    x = dense_two.forward(x)
    let loss = x.compute_loss["mse"](true_vals)

    let initial_start = now()
    var epoch_start = now()
    let bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        for i in range(input.initialize["random_uniform", 0, 1]().capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        avg_loss += loss.forward_static()[0]
        loss.backward()
        loss.optimize["sgd", 0.01]()

        if epoch % every == 0:
            var bar = String("")
            for i in range(bar_accuracy):
                if i < ((epoch * bar_accuracy) / num_epochs).to_int():
                    bar += "█"
                else:
                    bar += "░"
            clear()
            print_no_newline("\nEpoch: " + String(epoch) + " ")
            info(bar + " ")
            print_no_newline(String(((epoch * 100) / num_epochs).to_int()) + "%\n")
            print("----------------------------------------\n")
            print_no_newline("Average Loss: ")
            info(String(avg_loss / every) + "\n")
            print_no_newline("Time: ")
            info(String(nanoseconds_to_seconds(now() - epoch_start)) + "s\n")
            epoch_start = now()
            print("\n----------------------------------------\n")
            avg_loss = 0.0

    print_no_newline("Total Time: ")
    info(String(nanoseconds_to_seconds(now() - initial_start)) + "s\n\n")
    sys.external_call["exit", NoneType]()
