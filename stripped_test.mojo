from voodoo import Tensor
from voodoo.utils.shape import shape
from voodoo.layers.Dense import Dense
from benchmark import benchmark


alias data_shape = shape(32, 1)


fn main() raises:
    let W1 = Tensor(shape(1, 64)).initialize["he_normal"]()
    let W2 = Tensor(shape(64, 64)).initialize["he_normal"]()
    let W3 = Tensor(shape(64, 1)).initialize["he_normal"]()

    let b1 = Tensor(shape(64)).initialize["he_normal"]()
    let b2 = Tensor(shape(64)).initialize["he_normal"]()
    let b3 = Tensor(shape(1)).initialize["he_normal"]()

    let every = 2500
    let num_epochs = 50000

    let input = Tensor(data_shape).initialize["he_normal", 0, 1]().dynamic()
    let true_vals = Tensor(data_shape)

    var x = (input @ W1 + b1).compute_activation["relu"]()
    x = (x @ W2 + b2).compute_activation["relu"]()
    x = x @ W3 + b3
    let loss = x.compute_loss["mse"](true_vals)

    let bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        for i in range(input.initialize["random_uniform", 0, 1]().capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        _ = loss.forward_static()[0]
        loss.backward()
        loss.optimize["sgd", 0.01]()
