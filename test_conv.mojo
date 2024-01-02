from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    # TODO: Fix this (Specifically channels and the different types of layers that can be implemented)
    let pool = Layer[type="maxpool2d", in_neurons=1, out_neurons=1, pool_size=2, stride=2, padding=0]()
    let dropout = Layer[type="dropout", in_neurons=1, out_neurons=1, dropout_rate=0.5]()
    let flatten = Layer[type="flatten", in_neurons=1, out_neurons=1]()
    let dense = Layer[type="dense", in_neurons=1, out_neurons=1, activation="relu"]()
    let dropout2 = Layer[type="dropout", in_neurons=1, out_neurons=1, dropout_rate=0.5]()
    let output_layer = Layer[type="dense", in_neurons=1, out_neurons=1, activation="softmax"]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(shape(1, 28, 28))
    let true_vals = Tensor(shape(10))
    
    var x = pool.forward(input)
    x = dropout.forward(x)
    x = flatten.forward(x)
    x = dense.forward(x)
    x = dropout2.forward(x)
    x = output_layer.forward(x)
    let loss = x.compute_loss[get_loss_code["cce"]()](true_vals)
    
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
        loss.optimize["sgd", 0.01]()

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
