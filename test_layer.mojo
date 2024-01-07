from voodoo import Tensor, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from voodoo.layers.LeakyReLu import LeakyReLu
from voodoo.layers.Dense import Dense
from voodoo.layers.Dropout import Dropout
from time.time import now

fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    let input_layer = LeakyReLu[in_neurons=1, out_neurons=64]()
    let dense_layer = Dense[in_neurons=64, out_neurons=64, activation="relu"]()
    let output_layer = Dense[in_neurons=64, out_neurons=1]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(shape(32, 1)).initialize["he_normal", 0, 1]().dynamic()
    let true_vals = Tensor(shape(32, 1))

    # TODO, make a model struct to encapsulate this, variable n middle layers / total loss
    var x = input_layer.forward(input)
    x = dense_layer.forward(x)
    x = output_layer.forward(x)
    let loss = x.compute_loss["mse"](true_vals)

    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        for i in range(input.initialize["random_uniform", 0, 1]().capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        avg_loss += loss.forward_static()[0]
        loss.backward()
        loss.optimize["adam", 0.01]()

        if epoch % every == 0:
            print_no_newline(chr(27) + "[2J")
            print()
            print_no_newline(
                String("Epoch:") +
                epoch +
                " Avg Loss: " +
                avg_loss / every +
                " Time: " +
                nanoseconds_to_seconds(now() - epoch_start) +
                "s"
            )
            avg_loss = 0.0

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
