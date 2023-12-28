from voodoo.losses import (
    mae,
    mape,
    mse,
    msle,
    bce,
    cce,
    cfce,
)

from voodoo import Tensor

# Data from Tensorflow


fn copyTensor(t: Tensor) raises -> Tensor:
    let shape = t.node_ptr.load().load().shape_ptr.load()
    let newTensor = Tensor(shape)
    for i in range(0, t.capacity()):
        newTensor[i] = t[i]
    return newTensor


fn is_close(a: Float32, b: Float32) -> Bool:
    if a > b:
        return a / b < 1.01
    else:
        return b / a < 1.01


fn passed(
    expected_s: FloatLiteral,
    expected_m: FloatLiteral,
    expected_l: FloatLiteral,
    predicted_s: Tensor,
    predicted_m: Tensor,
    predicted_l: Tensor,
) raises -> Bool:
    let result = (
        is_close(expected_s, predicted_s[0])
        and is_close(expected_m, predicted_m[0])
        and is_close(expected_l, predicted_l[0])
    )
    if not result:
        print("Expected: ")
        print(expected_s)
        print(expected_m)
        print(expected_l)
        print("Got: ")
        predicted_s.print()
        predicted_m.print()
        predicted_l.print()
    return result


fn print_bars():
    print("--------------------------------------------------")


fn test_mae(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = 1.0268464088439941
    let expected_m = 2.0149545669555664
    let expected_l = 3.984262466430664
    let predicted_s = mae(s_p, s_t)
    let predicted_m = mae(m_p, m_t)
    let predicted_l = mae(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_mape(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = 89312.0390625
    let expected_m = 6966669.5
    let expected_l = 5252788.0
    let predicted_s = mape(s_p, s_t)
    let predicted_m = mape(m_p, m_t)
    let predicted_l = mape(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_mse(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = 1.5469061136245728
    let expected_m = 5.558147430419922
    let expected_l = 21.332069396972656
    let predicted_s = mse(s_p, s_t)
    let predicted_m = mse(m_p, m_t)
    let predicted_l = mse(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_msle(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = 0.1217174157500267
    let expected_m = 0.4461246132850647
    let expected_l = 1.047558307647705
    let predicted_s = msle(s_p, s_t)
    let predicted_m = msle(m_p, m_t)
    let predicted_l = msle(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_bce(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = -4.433718681335449
    let expected_m = -8.112979888916016
    let expected_l = -11.845508575439453
    let predicted_s = bce(s_p, s_t)
    let predicted_m = bce(m_p, m_t)
    let predicted_l = bce(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_cce(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = -16.11379623413086
    let expected_m = -0.5549430847167969
    let expected_l = -11.89748764038086
    let predicted_s = cce(s_p, s_t)
    let predicted_m = cce(m_p, m_t)
    let predicted_l = cce(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_cfce(
    s_p: Tensor, m_p: Tensor, l_p: Tensor, s_t: Tensor, m_t: Tensor, l_t: Tensor
) raises -> Bool:
    let expected_s = -4.029522895812988
    let expected_m = -0.03908872604370117
    let expected_l = -3.016561269760132
    let predicted_s = cfce(s_p, s_t)
    let predicted_m = cfce(m_p, m_t)
    let predicted_l = cfce(l_p, l_t)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn main() raises:
    var vals = DynamicVector[Float32]()
    var shape = DynamicVector[Int]()

    var predictionTensor1Data = DynamicVector[Float32]()
    predictionTensor1Data.append(-0.017705373466014862)
    predictionTensor1Data.append(0.08340083807706833)
    predictionTensor1Data.append(-0.00035693656536750495)
    predictionTensor1Data.append(-0.04133322089910507)

    var predictionTensor2Data = DynamicVector[Float32]()
    predictionTensor2Data.append(-0.010720985010266304)
    predictionTensor2Data.append(0.08854445070028305)
    predictionTensor2Data.append(-0.010992827825248241)
    predictionTensor2Data.append(-0.0139495013281703)
    predictionTensor2Data.append(-0.05573265627026558)
    predictionTensor2Data.append(0.022942548617720604)
    predictionTensor2Data.append(-0.03994018957018852)
    predictionTensor2Data.append(0.005975843872874975)

    var predictionTensor3Data = DynamicVector[Float32]()
    predictionTensor3Data.append(0.04057735577225685)
    predictionTensor3Data.append(0.0013483938528224826)
    predictionTensor3Data.append(-0.06115312501788139)
    predictionTensor3Data.append(0.06560912728309631)
    predictionTensor3Data.append(-0.12968583405017853)
    predictionTensor3Data.append(-0.05152547359466553)
    predictionTensor3Data.append(0.025589486584067345)
    predictionTensor3Data.append(0.034501541405916214)
    predictionTensor3Data.append(0.08404312282800674)
    predictionTensor3Data.append(0.01694565825164318)
    predictionTensor3Data.append(0.005890246015042067)
    predictionTensor3Data.append(0.1145363375544548)
    predictionTensor3Data.append(0.04264656826853752)
    predictionTensor3Data.append(0.013216649182140827)
    predictionTensor3Data.append(0.004337237682193518)
    predictionTensor3Data.append(0.063532255589962)

    shape.append(2)
    shape.append(2)

    for i in range(-2, 2):
        vals.append(i)

    let testTensor1 = Tensor(shape)._custom_fill(vals)
    let predictionTensor1 = Tensor(shape)._custom_fill(predictionTensor1Data)

    shape.append(2)
    vals.clear()
    for i in range(-4, 4):
        vals.append(i)

    let testTensor2 = Tensor(shape)._custom_fill(vals)
    let predictionTensor2 = Tensor(shape)._custom_fill(predictionTensor2Data)

    shape.append(2)
    vals.clear()
    for i in range(-8, 8):
        vals.append(i)

    let testTensor3 = Tensor(shape)._custom_fill(vals)
    let predictionTensor3 = Tensor(shape)._custom_fill(predictionTensor3Data)

    var predictionTensorShape = DynamicVector[Int]()
    predictionTensorShape.append(2)
    predictionTensorShape.append(2)

    print_bars()
    print("Testing MSE")
    print_bars()
    if test_mse(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("MSE test passed")
    else:
        print("MSE test failed")

    print_bars()
    print("Testing MAE")
    print_bars()
    if test_mae(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("MAE test passed")
    else:
        print("MAE test failed")

    print_bars()
    print("Testing MAPE")
    print_bars()
    if test_mape(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("MAPE test passed")
    else:
        print("MAPE test failed")

    print_bars()
    print("Testing MSE")
    print_bars()
    if test_mse(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("MSE test passed")
    else:
        print("MSE test failed")

    print_bars()
    print("Testing MSLE")
    print_bars()
    if test_msle(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("MSLE test passed")
    else:
        print("MSLE test failed")

    print_bars()
    print("Testing BCE")
    print_bars()
    if test_bce(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("BCE test passed")
    else:
        print("BCE test failed")

    print_bars()
    print("Testing CCE")
    print_bars()
    if test_cce(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("CCE test passed")
    else:
        print("CCE test failed")

    print_bars()
    print("Testing CFCE")
    print_bars()
    if test_cfce(
        copyTensor(predictionTensor1),
        copyTensor(predictionTensor2),
        copyTensor(predictionTensor3),
        copyTensor(testTensor1),
        copyTensor(testTensor2),
        copyTensor(testTensor3),
    ):
        print("CFCE test passed")
    else:
        print("CFCE test failed")
