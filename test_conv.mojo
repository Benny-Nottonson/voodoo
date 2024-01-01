from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    # TODO: Fix this (Specifically channels and the different types of layers that can be implemented)
    let input_layer = Layer[type="conv2d", activation="relu", kernel_width=3, kernel_height=3](1, 64)
    let hidden_layer = Layer[type="conv2d", activation="relu", kernel_width=3, kernel_height=3](64, 64)
    let pool = Layer[type="maxpool2d", pool_size=2](64, 64)
    let dropout = Layer[type="dropout", dropout_rate=0.25](64, 64)
    let flatten = Layer[type="flatten"](64, 64)
    let dense = Layer[type="dense", activation="relu"](64, 128)
    let dropout2 = Layer[type="dropout", dropout_rate=0.5](128, 128)
    let output_layer = Layer[type="dense", activation="softmax"](128, 10)

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(shape(1, 28, 28))
    let true_vals = Tensor(shape(10))
    
    var x = input_layer.forward(input)
    x = hidden_layer.forward(x)
    x = pool.forward(x)
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
