from voodoo import Tensor, Dense, mse
from voodoo.utils.shape import shape
from time.time import now
from math import max, min


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return Float64(t) / 1_000_000_000.0


alias loss_fn = mse


fn main() raises:
    var min_temps = DynamicVector[Float32]()
    var max_temps = DynamicVector[Float32]()
    with open("weather.csv", "r") as f:
        let data = f.read().split("\n")
        for line in range(1, len(data) - 1):
            let line_data = data[line].split(",")
            let max_temp = line_data[4].split(".")
            let min_temp = line_data[5].split(".")
            if len(max_temp) == 1:
                max_temps.append(atol(max_temp[0]))
            else:
                max_temps.append(
                    atol(max_temp[0]) + atol(max_temp[1]) / 10.0 ** len(max_temp[1])
                )

            if len(min_temp) == 1:
                min_temps.append(atol(min_temp[0]))
            else:
                min_temps.append(
                    atol(min_temp[0]) + atol(min_temp[1]) / 10.0 ** len(min_temp[1])
                )

    let train_size = 95232
    let test_size = 23808

    var max_temp: Float32 = 0.0
    var min_temp: Float32 = 100.0
    for i in range(train_size):
        max_temp = max(max_temp, max_temps[i])
        min_temp = min(min_temp, min_temps[i])

    for i in range(train_size):
        max_temps[i] = (max_temps[i] - min_temp) / (max_temp - min_temp)
        min_temps[i] = (min_temps[i] - min_temp) / (max_temp - min_temp)

    let input_layer = Dense[activation="relu"](1, 64)
    let dense_layer = Dense[activation="relu"](64, 64)
    let output_layer = Dense(64, 1)

    var avg_loss: Float32 = 0.0

    let every = 64
    let num_epochs = train_size / every

    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        let input = Tensor(shape(every, 1))
        let true_vals = Tensor(shape(every, 1))

        for i in range(every):
            let idx = (epoch - 1) * every + i
            input[i] = min_temps[idx]
            true_vals[i] = max_temps[idx]

        var x = input_layer.forward(input)
        x = dense_layer.forward(x)
        x = output_layer.forward(x)
        let loss = loss_fn(x, true_vals)

        avg_loss += loss[0]
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
        loss.optimize["sgd"](0.01)

        loss.clear()
        input.free()

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")

    # Now test the model and track the average loss (Same batch size)

    let num_epochs_test = test_size / every
    avg_loss = 0.0
    var test_loss: Float32 = 0.0
    for epoch in range(1, num_epochs_test + 1):
        let epoch_start = now()
        let input = Tensor(shape(every, 1))
        let true_vals = Tensor(shape(every, 1))

        for i in range(every):
            let idx = (epoch - 1) * every + i
            input[i] = min_temps[idx]
            true_vals[i] = max_temps[idx]

        var x = input_layer.forward(input)
        x = dense_layer.forward(x)
        x = output_layer.forward(x)
        let loss = loss_fn(x, true_vals)

        avg_loss += loss[0]
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
            test_loss += avg_loss / every
            avg_loss = 0.0

        loss.clear()
        input.free()

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
    print("Average Loss: ", test_loss / num_epochs_test.cast[DType.float32]())
