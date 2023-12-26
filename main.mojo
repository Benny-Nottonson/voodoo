from models import BasicCNN
from utilities import Matrix

fn main():
    var testModel = BasicCNN[
        name="testModel",
        input_dim=784,
        output_dim=10
    ]()

    for i in range(10):
        let x = Matrix.rand(784, 1)
        let y = testModel.forward(x)
        print(y)