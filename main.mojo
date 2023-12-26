from models import BasicCNN
from utilities import Matrix

fn main():
    let testModel = BasicCNN[
        name="testModel",
        input_dim=784,
        output_dim=10
    ]()