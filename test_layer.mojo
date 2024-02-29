from time.time import now
from tensor import TensorShape

from voodoo.core import Tensor, HeNormal, RandomUniform, SGD
from voodoo.core.layers import Dense, LeakyReLu, Dropout
from voodoo.utils import (
    info,
    clear,
)


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    var input_layer = Dense[
        in_neurons=1,
        out_neurons=32,
        activation="relu",
        weight_initializer = HeNormal[1],
        bias_initializer = HeNormal[32],
    ]()
    var dropout = Dropout[dropout_rate=0.01,]()
    var leaky_relu = LeakyReLu[
        in_neurons=32,
        out_neurons=32,
        weight_initializer = HeNormal[32],
        bias_initializer = HeNormal[32],
    ]()
    var output_layer = Dense[
        in_neurons=32,
        out_neurons=1,
        weight_initializer = HeNormal[32],
        bias_initializer = HeNormal[1],
    ]()

    var avg_loss: Float32 = 0.0
    var every = 1000
    var num_epochs = 2000000

    var input = Tensor[TensorShape(32, 1), RandomUniform[0, 1]]()
    var true_vals = Tensor[TensorShape(32, 1), RandomUniform[0, 1]]()

    var x0 = input_layer.forward(input)
    var x1 = dropout.forward(x0)
    var x2 = leaky_relu.forward(x1)
    var x3 = output_layer.forward(x2)
    var loss = x3.compute_loss["mse"](true_vals)

    var initial_start = now()
    var epoch_start = now()
    var bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        input.refresh()
        for i in range(input.shape.num_elements()):
            true_vals[i] = math.sin(15.0 * input[i])

        var computed_loss = loss.forward_static()
        avg_loss += computed_loss[0]
        loss.backward()
        loss.optimize[SGD[0.01]]()

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
