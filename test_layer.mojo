from voodoo import Tensor, get_loss_code, Graph
from voodoo.layers.Dense import Dense
from voodoo.utils import (
    info,
    clear,
)
from time.time import now
from tensor import TensorShape


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


alias data_shape = TensorShape(32, 1)


fn main() raises:
    let input_layer = Dense[
        in_neurons=1, out_neurons=64, activation="relu", bias_initializer="he_normal"
    ]()
    let dense_layer = Dense[
        in_neurons=64, out_neurons=64, activation="relu", bias_initializer="he_normal"
    ]()
    let output_layer = Dense[
        in_neurons=64, out_neurons=1, bias_initializer="he_normal"
    ]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    var input = Tensor(data_shape).initialize["he_normal", 0, 1]()
    input = input
    let true_vals = Tensor(data_shape)

    var x = input_layer.forward(input)
    x = dense_layer.forward(x)
    x = output_layer.forward(x)
    var loss = x.compute_loss["mse"](true_vals)

    let initial_start = now()
    var epoch_start = now()
    let bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        for i in range(input.initialize["random_uniform", 0, 1]().node.cap_ptr.load()):
            true_vals[i] = math.sin(15.0 * input[i])

        var computed_loss = loss.forward_static()
        avg_loss += computed_loss[0]
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
