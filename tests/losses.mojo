from python import Python

from voodoo import Tensor, get_loss_code

alias TestSize = 5


fn test_fn[
    f: String
](
    tfFunction: PythonObject,
    tfConstant: PythonObject,
    tfType: PythonObject,
    tfSum: PythonObject,
) raises -> Int:
    var smallShape = DynamicVector[Int]()
    var mediumShape = DynamicVector[Int]()
    var largeShape = DynamicVector[Int]()

    smallShape.append(TestSize)
    smallShape.append(TestSize)
    mediumShape.append(TestSize)
    mediumShape.append(TestSize)
    mediumShape.append(TestSize)
    largeShape.append(TestSize)
    largeShape.append(TestSize)
    largeShape.append(TestSize)
    largeShape.append(TestSize)

    let smallTensorInitial = Tensor(smallShape).initialize["random_normal"]()
    let mediumTensorInitial = Tensor(mediumShape).initialize["random_normal"]()
    let largeTensorInitial = Tensor(largeShape).initialize["random_normal"]()

    let smallTensorGuess = Tensor(smallShape).initialize["random_normal"]()
    let mediumTensorGuess = Tensor(mediumShape).initialize["random_normal"]()
    let largeTensorGuess = Tensor(largeShape).initialize["random_normal"]()

    let smallTensorLoss = smallTensorInitial.compute_loss[get_loss_code[f]()](
        smallTensorGuess
    )
    let mediumTensorLoss = mediumTensorInitial.compute_loss[get_loss_code[f]()](
        mediumTensorGuess
    )
    let largeTensorLoss = largeTensorInitial.compute_loss[get_loss_code[f]()](
        largeTensorGuess
    )

    let smallTest: PythonObject = []
    let mediumTest: PythonObject = []
    let largeTest: PythonObject = []

    let smallTestTwo: PythonObject = []
    let mediumTestTwo: PythonObject = []
    let largeTestTwo: PythonObject = []

    let smallGuess: PythonObject = []
    let mediumGuess: PythonObject = []
    let largeGuess: PythonObject = []

    for i in range(TestSize**2):
        _ = smallTest.append(smallTensorInitial[i])
        _ = smallTestTwo.append(smallTensorGuess[i])
        _ = smallGuess.append(smallTensorLoss[i])

    for i in range(TestSize**3):
        _ = mediumTest.append(mediumTensorInitial[i])
        _ = mediumTestTwo.append(mediumTensorGuess[i])
        _ = mediumGuess.append(mediumTensorLoss[i])

    for i in range(TestSize**4):
        _ = largeTest.append(largeTensorInitial[i])
        _ = largeTestTwo.append(largeTensorGuess[i])
        _ = largeGuess.append(largeTensorLoss[i])

    let resultSmall = tfFunction(
        tfConstant(smallTest, tfType, [TestSize, TestSize]),
        tfConstant(smallTestTwo, tfType, [TestSize, TestSize]),
    )
    let resultMedium = tfFunction(
        tfConstant(mediumTest, tfType, [TestSize, TestSize, TestSize]),
        tfConstant(mediumTestTwo, tfType, [TestSize, TestSize, TestSize]),
    )
    let resultLarge = tfFunction(
        tfConstant(largeTest, tfType, [TestSize, TestSize, TestSize, TestSize]),
        tfConstant(largeTestTwo, tfType, [TestSize, TestSize, TestSize, TestSize]),
    )

    let mojoResultSmall = tfConstant(smallGuess, tfType, [TestSize, TestSize])
    let mojoResultMedium = tfConstant(
        mediumGuess, tfType, [TestSize, TestSize, TestSize]
    )
    let mojoResultLarge = tfConstant(
        largeGuess, tfType, [TestSize, TestSize, TestSize, TestSize]
    )

    let resSmall: Bool = (
        tfSum(resultSmall.__abs__() - mojoResultSmall.__abs__()) < 0.05
    ).numpy().__bool__()
    let resMedium: Bool = (
        tfSum(resultMedium.__abs__() - mojoResultMedium.__abs__()) < 0.05
    ).numpy().__bool__()
    let resLarge: Bool = (
        tfSum(resultLarge.__abs__() - mojoResultLarge.__abs__()) < 0.05
    ).numpy().__bool__()

    print("----- Test for " + tfFunction.__name__.__str__() + " -----")
    if resSmall:
        print("✅ Small test passed")
    else:
        print("❌ Small test failed")

    if resMedium:
        print("✅ Medium test passed")
    else:
        print("❌ Medium test failed")

    if resLarge:
        print("✅ Large test passed")
    else:
        print("❌ Large test failed")

    if resSmall and resMedium and resLarge:
        print("----- All tests passed -----")
    else:
        print("----- Some tests failed -----")

    print("---------------------------------")

    var failed = 0
    if not resSmall:
        failed += 1
    if not resMedium:
        failed += 1
    if not resLarge:
        failed += 1

    return failed


fn main() raises:
    let tf = Python.import_module("tensorflow")
    var total = 0

    total += test_fn["mae"](
        tf.keras.losses.mae, tf.constant, tf.float32, tf.math.reduce_sum
    )
    total += test_fn["mape"](
        tf.keras.losses.mape, tf.constant, tf.float32, tf.math.reduce_sum
    )
    total += test_fn["mse"](
        tf.keras.losses.mse, tf.constant, tf.float32, tf.math.reduce_sum
    )
    total += test_fn["msle"](
        tf.keras.losses.msle, tf.constant, tf.float32, tf.math.reduce_sum
    )
    total += test_fn["bce"](
        tf.keras.losses.binary_crossentropy,
        tf.constant,
        tf.float32,
        tf.math.reduce_sum,
    )
    total += test_fn["cce"](
        tf.keras.losses.categorical_crossentropy,
        tf.constant,
        tf.float32,
        tf.math.reduce_sum,
    )
    total += test_fn["cfce"](
        tf.keras.losses.categorical_hinge,
        tf.constant,
        tf.float32,
        tf.math.reduce_sum,
    )

    if total == 0:
        print("✅ All tests passed")
    elif total == 1:
        print("❌ ", total, " test failed")
    else:
        print("❌ ", total, " tests failed")
