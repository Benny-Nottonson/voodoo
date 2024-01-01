from voodoo import Tensor, Layer, sin, get_loss_code, Graph
from voodoo.utils.shape import shape
from time.time import now


fn nanoseconds_to_seconds(t: Int) -> Float64:
    return t / 1_000_000_000.0


fn main() raises:
    """
    TODO:
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(28,28,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    This should be added once Dropout, Flatten, and MaxPooling layers have been added
    """

    var avg_loss: Float32 = 0.0
    let every = 1000
    let num_epochs = 20000

    let initial_start = now()
    for epoch in range(1, num_epochs + 1):
        let epoch_start = now()

    print("Total Time: ", nanoseconds_to_seconds(now() - initial_start), "s")
