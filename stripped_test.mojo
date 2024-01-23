from voodoo import Tensor
from voodoo.utils.shape import shape
from voodoo.layers.Dense import Dense
from benchmark import benchmark


alias data_shape = shape(32, 1)


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

    let every = 2500
    let num_epochs = 50000

    let input = Tensor(data_shape).initialize["he_normal", 0, 1]().dynamic()
    let true_vals = Tensor(data_shape)

    var x = input_layer.forward(input)
    x = dense_layer.forward(x)
    x = output_layer.forward(x)
    let loss = x.compute_loss["mse"](true_vals)

    let bar_accuracy = 20
    for epoch in range(1, num_epochs + 1):
        for i in range(input.initialize["random_uniform", 0, 1]().capacity()):
            true_vals[i] = math.sin(15.0 * input[i])

        _=loss.forward_static()[0]
        loss.backward()
        loss.optimize["sgd", 0.01]()