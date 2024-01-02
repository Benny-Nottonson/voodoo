from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    let input_layer = Layer[type="dense", in_neurons=1, out_neurons=64, activation="elu"]()
    let dense_layer = Layer[type="leaky_relu", in_neurons=64, out_neurons=64]()
    let output_layer = Layer[type="dense", in_neurons=64, out_neurons=1]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(shape(32, 1)).initialize["he_normal", 0, 1]()
    let true_vals = Tensor(shape(32, 1))

    # TODO, make a model struct to encapsulate this, variable n middle layers / total loss
    var x = input_layer.forward(input)
    x = dense_layer.forward(x)
    x = output_layer.forward(x)
    let loss = x.compute_loss[get_loss_code["mse"]()](true_vals)

    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        for i in range(input.initialize["random_uniform", 0, 1]().capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        avg_loss += loss.forward_static()[0]
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

        loss.backward()
        loss.optimize["adam", 0.01]()

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
