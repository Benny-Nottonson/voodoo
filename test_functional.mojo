from voodoo import (
    Tensor,
    sin,
    relu,
    mse,
)
from voodoo.utils.shape import shape
from time.time import now

fn nanoseconds_to_seconds(t: Int) -> Float64:
    return Float64(t) / 1_000_000_000.0
    

fn main() raises:
    let W1 = Tensor(shape(1, 64)).he_normal().requires_grad()
    let W2 = Tensor(shape(64, 64)).he_normal().requires_grad()
    let W3 = Tensor(shape(64, 1)).he_normal().requires_grad()
    
    let b1 = Tensor(shape(64)).he_normal().requires_grad()
    let b2 = Tensor(shape(64)).he_normal().requires_grad()
    let b3 = Tensor(shape(1)).he_normal().requires_grad()

    var avg_loss = Float32(0.0)
    let every = 1000
    let num_epochs = 20000

    let input = Tensor(shape(32, 1)).random_uniform(0, 1)
    let true_vals = Tensor(shape(32, 1))

    var x = relu(input @ W1 + b1)
    x = relu(x @ W2 + b2)
    x = x @ W3 + b3
    let loss = mse(x, true_vals)

    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        for i in range(input.random_uniform(0, 1).capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        avg_loss += loss.forward_static()[0]
        if epoch % every == 0:
            print("Epoch:", epoch, " Avg Loss: ", avg_loss / every, " Time: " , nanoseconds_to_seconds(now() - epoch_start), "s")
            avg_loss = 0.0

        loss.backward()
        loss.optimize["sgd"](0.01)

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")