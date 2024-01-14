from voodoo import (
    Tensor,
    get_activation_code,
    get_loss_code,
)
from voodoo.utils import (
    info,
    clear,
)
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return Float64(t) / 1_000_000_000.0


alias data_shape = shape(4, 1)


fn main() raises:
    let W1 = Tensor(shape(1, 64)).initialize["he_normal"]()
    let W2 = Tensor(shape(64, 64)).initialize["he_normal"]()
    let W3 = Tensor(shape(64, 1)).initialize["he_normal"]()

    let b1 = Tensor(shape(64)).initialize["he_normal"]()
    let b2 = Tensor(shape(64)).initialize["he_normal"]()
    let b3 = Tensor(shape(1)).initialize["he_normal"]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(data_shape).initialize["he_normal", 0, 1]().dynamic()
    let true_vals = Tensor(data_shape)

    var x = (input @ W1 + b1).compute_activation["relu"]()
    x = (x @ W2 + b2).compute_activation["relu"]()
    x = x @ W3 + b3
    let loss = x.compute_loss["mse"](true_vals)

    let initial_start = now()
    let bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
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
            print("\n----------------------------------------\n")
            avg_loss = 0.0

    print_no_newline("Total Time: ")
    info(String(nanoseconds_to_seconds(now() - initial_start)) + "s\n\n")
