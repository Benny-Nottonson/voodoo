from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now
from datasets import DataLoader


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    var data = DataLoader["mnist"]()

    let conv1 = Layer[type="conv2d", in_width=28, in_height=28, in_batches=32, kernel_width=5, kernel_height=5, stride=1, padding=0]()
    let max_pool1 = Layer[type="maxpool2d", pool_size=2, stride=2, padding=0]()
    let conv2 = Layer[type="conv2d", in_width=12, in_height=12, in_batches=32, kernel_width=5, kernel_height=5, stride=1, padding=0]()
    let max_pool2 = Layer[type="maxpool2d", pool_size=2, stride=2, padding=0]()
    let flatten = Layer[type="flatten"]()
    let dense1 = Layer[type="dense", in_neurons=16, out_neurons=16, activation="relu"]()
    let dropout = Layer[type="dropout", dropout_rate=0.1]()
    let dense2 = Layer[type="dense", in_neurons=16, out_neurons=10, activation="softmax"]()

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let firstData = data.load_data_as_tensor(32, 0) 
    let firstLabel = data.load_labels_as_tensor(32, 0)
    
    var x = conv1.forward(firstData)
    x = max_pool1.forward(x)
    x = conv2.forward(x)
    x = max_pool2.forward(x)
    x = flatten.forward(x)
    x = dense1.forward(x)
    x = dropout.forward(x)
    x = dense2.forward(x)
    let loss = x.compute_loss[get_loss_code["cce"]()](firstLabel)

    loss.print()
    
    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()
        
        for i in range(0, 32):
            let test_data = data.load_data_as_tensor(32, i) 
            let label = data.load_labels_as_tensor(32, i)

        avg_loss += loss.forward_static()[0]
        loss.backward()
        loss.optimize["sgd", 0.01]()

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

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
