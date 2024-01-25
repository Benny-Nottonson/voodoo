from voodoo import Tensor, get_loss_code, Graph
from voodoo.utils.shape import shape
from voodoo.layers.Dense import Dense
from voodoo.layers.Conv2D import Conv2D
from voodoo.layers.Reshape import Reshape
from voodoo.layers.MaxPool2D import MaxPool2D
from voodoo.utils import (
    info,
    clear,
)
from time.time import now
from datasets import MNist


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


alias batches = 32
alias channels = 1
alias width = 28
alias height = 28

alias data_shape = shape(batches, channels, width, height)


fn main() raises:
    let dataset = MNist()

    let conv_layer_one = Conv2D[
        in_channels=1,
        kernel_width=3,
        kernel_height=3,
        stride=1,
        padding=0,
        weight_initializer="he_uniform",
        activation="relu",
    ]()
    let max_pool_one = MaxPool2D[
        kernel_width=2,
        kernel_height=2,
        stride=2,
    ]()
    let reshape = Reshape[shape(batches, 169)]()
    let dense1 = Dense[
        in_neurons=169,
        out_neurons=100,
        weight_initializer="he_uniform",
        activation="relu",
    ]()
    let dense2 = Dense[
        in_neurons=100,
        out_neurons=10,
        activation="sigmoid",
    ]()

    var avg_loss: Float32 = 0.0
    let every = 90
    let num_epochs = 1800

    let true_vals = Tensor(shape(batches, 10)).initialize["zeros"]().dynamic()
    let input = Tensor(shape(batches, channels, width, height)).initialize[
        "zeros"
    ]().dynamic()

    for i in range(batches):
        let image = dataset.train_images[i]
        let label = dataset.train_labels[i].to_int()
        true_vals[i * 10 + label] = 1.0
        for j in range(width):
            for k in range(height):
                input[i * channels * width * height + j * width + k] = image[
                    j * width + k
                ].to_int()

    var x = conv_layer_one.forward(input)
    x = max_pool_one.forward(x)
    x = reshape.forward(x)
    x = dense1.forward(x)
    x = dense2.forward(x)
    let loss = x.compute_loss["mse"](true_vals)

    let initial_start = now()
    var epoch_start = now()
    let bar_accuracy = 20

    for epoch in range(1, num_epochs + 1):
        for i in range(batches):
            let image = dataset.train_images[i + epoch * batches]
            let label = dataset.train_labels[i + epoch * batches].to_int()
            true_vals[i * 10 + label] = 1.0
            for j in range(width):
                for k in range(height):
                    input[i * channels * width * height + j * width + k] = image[
                        j * width + k
                    ].to_int()

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
    external_call["exit", NoneType]()
