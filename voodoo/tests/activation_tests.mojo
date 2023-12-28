from voodoo.activations import (
    elu,
    exp,
    gelu,
    hard_sigmoid,
    linear,
    mish,
    relu,
    selu,
    sigmoid,
    softmax,
    softplus,
    softsign,
    swish,
    tanh,
)

from voodoo import Tensor

# Data from Tensorflow


fn is_close(a: Float32, b: Float32) -> Bool:
    let diff = a - b
    if diff < 0.0:
        return diff > -0.0001
    else:
        return diff < 0.0001


fn passed(
    expected_s: ListLiteral[FloatLiteral, FloatLiteral, FloatLiteral, FloatLiteral],
    expected_m: ListLiteral[
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
    ],
    expected_l: ListLiteral[
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
        FloatLiteral,
    ],
    predicted_s: Tensor,
    predicted_m: Tensor,
    predicted_l: Tensor,
) raises -> Bool:
    let result = (
        is_close(expected_s.get[0, FloatLiteral](), predicted_s[0])
        and is_close(expected_s.get[1, FloatLiteral](), predicted_s[1])
        and is_close(expected_s.get[2, FloatLiteral](), predicted_s[2])
        and is_close(expected_s.get[3, FloatLiteral](), predicted_s[3])
        and is_close(expected_m.get[0, FloatLiteral](), predicted_m[0])
        and is_close(expected_m.get[1, FloatLiteral](), predicted_m[1])
        and is_close(expected_m.get[2, FloatLiteral](), predicted_m[2])
        and is_close(expected_m.get[3, FloatLiteral](), predicted_m[3])
        and is_close(expected_m.get[4, FloatLiteral](), predicted_m[4])
        and is_close(expected_m.get[5, FloatLiteral](), predicted_m[5])
        and is_close(expected_m.get[6, FloatLiteral](), predicted_m[6])
        and is_close(expected_m.get[7, FloatLiteral](), predicted_m[7])
        and is_close(expected_l.get[0, FloatLiteral](), predicted_l[0])
        and is_close(expected_l.get[1, FloatLiteral](), predicted_l[1])
        and is_close(expected_l.get[2, FloatLiteral](), predicted_l[2])
        and is_close(expected_l.get[3, FloatLiteral](), predicted_l[3])
        and is_close(expected_l.get[4, FloatLiteral](), predicted_l[4])
        and is_close(expected_l.get[5, FloatLiteral](), predicted_l[5])
        and is_close(expected_l.get[6, FloatLiteral](), predicted_l[6])
        and is_close(expected_l.get[7, FloatLiteral](), predicted_l[7])
        and is_close(expected_l.get[8, FloatLiteral](), predicted_l[8])
        and is_close(expected_l.get[9, FloatLiteral](), predicted_l[9])
        and is_close(expected_l.get[10, FloatLiteral](), predicted_l[10])
        and is_close(expected_l.get[11, FloatLiteral](), predicted_l[11])
        and is_close(expected_l.get[12, FloatLiteral](), predicted_l[12])
        and is_close(expected_l.get[13, FloatLiteral](), predicted_l[13])
        and is_close(expected_l.get[14, FloatLiteral](), predicted_l[14])
        and is_close(expected_l.get[15, FloatLiteral](), predicted_l[15])
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


fn test_elu(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-0.8646647334098816, -0.6321205496788025, 0.0, 1.0]
    let expected_m = [
        -0.9816843867301941,
        -0.9502129554748535,
        -0.8646647334098816,
        -0.6321205496788025,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    let expected_l = [
        -0.9996645450592041,
        -0.9990881085395813,
        -0.9975212216377258,
        -0.9932620525360107,
        -0.9816843867301941,
        -0.9502129554748535,
        -0.8646647334098816,
        -0.6321205496788025,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    let predicted_s = elu(s)
    let predicted_m = elu(m)
    let predicted_l = elu(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_exp(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [0.1353352832366127, 0.3678794503211975, 1.0, 2.7182817459106445]
    let expected_m = [
        0.018315639346837997,
        0.049787070602178574,
        0.1353352814912796,
        0.3678794503211975,
        1.0,
        2.7182817459106445,
        7.389056205749512,
        20.08553695678711,
    ]
    let expected_l = [
        0.000335462624207139,
        0.0009118819725699723,
        0.0024787522852420807,
        0.0067379469983279705,
        0.018315639346837997,
        0.049787070602178574,
        0.1353352814912796,
        0.3678794503211975,
        1.0,
        2.7182817459106445,
        7.389056205749512,
        20.08553695678711,
        54.598148345947266,
        148.4131622314453,
        403.42877197265625,
        1096.6331787109375,
    ]
    let predicted_s = exp(s)
    let predicted_m = exp(m)
    let predicted_l = exp(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_gelu(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [
        -0.04550027847290039,
        -0.15865525603294373,
        0.0,
        0.8413447141647339,
    ]
    let expected_m = [
        -0.0001264810562133789,
        -0.0040496885776519775,
        -0.04550027847290039,
        -0.15865525603294373,
        0.0,
        0.8413447141647339,
        1.9544997215270996,
        2.995950222015381,
    ]
    let expected_l = [
        -0.0,
        -0.0,
        -0.0,
        -1.6391277313232422e-06,
        -0.0001264810562133789,
        -0.0040496885776519775,
        -0.04550027847290039,
        -0.15865525603294373,
        0.0,
        0.8413447141647339,
        1.9544997215270996,
        2.995950222015381,
        3.999873638153076,
        4.999998092651367,
        6.0,
        7.0,
    ]
    let predicted_s = gelu(s)
    let predicted_m = gelu(m)
    let predicted_l = gelu(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_hard_sigmoid(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [0.09999999403953552, 0.30000001192092896, 0.5, 0.699999988079071]
    let expected_m = [
        0.0,
        0.0,
        0.09999999403953552,
        0.30000001192092896,
        0.5,
        0.699999988079071,
        0.8999999761581421,
        1.0,
    ]
    let expected_l = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.09999999403953552,
        0.30000001192092896,
        0.5,
        0.699999988079071,
        0.8999999761581421,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    let predicted_s = hard_sigmoid(s)
    let predicted_m = hard_sigmoid(m)
    let predicted_l = hard_sigmoid(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_linear(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-2.0, -1.0, 0.0, 1.0]
    let expected_m = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    let expected_l = [
        -8.0,
        -7.0,
        -6.0,
        -5.0,
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    let predicted_s = linear(s)
    let predicted_m = linear(m)
    let predicted_l = linear(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_mish(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-0.2525014579296112, -0.3034014403820038, 0.0, 0.8650983572006226]
    let expected_m = [
        -0.07259173691272736,
        -0.14564745128154755,
        -0.2525014579296112,
        -0.3034014403820038,
        0.0,
        0.8650983572006226,
        1.9439589977264404,
        2.986534833908081,
    ]
    let expected_l = [
        -0.002683250932022929,
        -0.006380262318998575,
        -0.014854080975055695,
        -0.03357623517513275,
        -0.07259173691272736,
        -0.14564745128154755,
        -0.2525014579296112,
        -0.3034014403820038,
        0.0,
        0.8650983572006226,
        1.9439589977264404,
        2.986534833908081,
        3.9974122047424316,
        4.999551773071289,
        5.999927043914795,
        6.999988555908203,
    ]
    let predicted_s = mish(s)
    let predicted_m = mish(m)
    let predicted_l = mish(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_relu(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [0.0, 0.0, 0.0, 1.0]
    let expected_m = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    let expected_l = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]
    let predicted_s = relu(s)
    let predicted_m = relu(m)
    let predicted_l = relu(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_selu(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-1.5201665163040161, -1.1113307476043701, 0.0, 1.0507010221481323]
    let expected_m = [
        -1.7258986234664917,
        -1.6705687046051025,
        -1.5201665163040161,
        -1.1113307476043701,
        0.0,
        1.0507010221481323,
        2.1014020442962646,
        3.1521029472351074,
    ]
    let expected_l = [
        -1.7575095891952515,
        -1.7564960718154907,
        -1.7537413835525513,
        -1.7462533712387085,
        -1.7258986234664917,
        -1.6705687046051025,
        -1.5201665163040161,
        -1.1113307476043701,
        0.0,
        1.0507010221481323,
        2.1014020442962646,
        3.1521029472351074,
        4.202804088592529,
        5.253505229949951,
        6.304205894470215,
        7.354907035827637,
    ]
    let predicted_s = selu(s)
    let predicted_m = selu(m)
    let predicted_l = selu(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_sigmoid(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [0.11920291930437088, 0.2689414322376251, 0.5, 0.7310585975646973]
    let expected_m = [
        0.01798621006309986,
        0.047425877302885056,
        0.11920291930437088,
        0.2689414322376251,
        0.5,
        0.7310585975646973,
        0.8807970881462097,
        0.9525741338729858,
    ]
    let expected_l = [
        0.00033535013790242374,
        0.0009110512328334153,
        0.0024726232513785362,
        0.006692850962281227,
        0.01798621006309986,
        0.047425877302885056,
        0.11920291930437088,
        0.2689414322376251,
        0.5,
        0.7310585975646973,
        0.8807970881462097,
        0.9525741338729858,
        0.9820137619972229,
        0.9933071732521057,
        0.9975273609161377,
        0.9990889430046082,
    ]
    let predicted_s = sigmoid(s)
    let predicted_m = sigmoid(m)
    let predicted_l = sigmoid(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_softmax(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
    ]
    let expected_m = [
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
    ]
    let expected_l = [
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
        0.2689414322376251,
        0.7310585379600525,
    ]
    let predicted_s = softmax(s)
    let predicted_m = softmax(m)
    let predicted_l = softmax(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_softplus(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [
        0.12692800164222717,
        0.3132616877555847,
        0.6931471824645996,
        1.31326162815094,
    ]
    let expected_m = [
        0.018149929121136665,
        0.04858735576272011,
        0.12692800164222717,
        0.3132616877555847,
        0.6931471824645996,
        1.31326162815094,
        2.1269280910491943,
        3.0485870838165283,
    ]
    let expected_l = [
        0.00033540636650286615,
        0.0009114663698710501,
        0.002475685440003872,
        0.006715348921716213,
        0.018149929121136665,
        0.04858735576272011,
        0.12692800164222717,
        0.3132616877555847,
        0.6931471824645996,
        1.31326162815094,
        2.1269280910491943,
        3.0485870838165283,
        4.0181498527526855,
        5.006715297698975,
        6.002475738525391,
        7.000911712646484,
    ]
    let predicted_s = softplus(s)
    let predicted_m = softplus(m)
    let predicted_l = softplus(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_softsign(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-0.6666666865348816, -0.5, 0.0, 0.5]
    let expected_m = [
        -0.800000011920929,
        -0.75,
        -0.6666666865348816,
        -0.5,
        0.0,
        0.5,
        0.6666666865348816,
        0.75,
    ]
    let expected_l = [
        -0.8888888955116272,
        -0.875,
        -0.8571428656578064,
        -0.8333333134651184,
        -0.800000011920929,
        -0.75,
        -0.6666666865348816,
        -0.5,
        0.0,
        0.5,
        0.6666666865348816,
        0.75,
        0.800000011920929,
        0.8333333134651184,
        0.8571428656578064,
        0.875,
    ]
    let predicted_s = softsign(s)
    let predicted_m = softsign(m)
    let predicted_l = softsign(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_swish(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [
        -0.23840583860874176,
        -0.2689414322376251,
        0.0,
        0.7310585975646973,
    ]
    let expected_m = [
        -0.07194484025239944,
        -0.14227762818336487,
        -0.23840583860874176,
        -0.2689414322376251,
        0.0,
        0.7310585975646973,
        1.7615941762924194,
        2.857722282409668,
    ]
    let expected_l = [
        -0.00268280110321939,
        -0.006377358455210924,
        -0.014835739508271217,
        -0.033464252948760986,
        -0.07194484025239944,
        -0.14227762818336487,
        -0.23840583860874176,
        -0.2689414322376251,
        0.0,
        0.7310585975646973,
        1.7615941762924194,
        2.857722282409668,
        3.9280550479888916,
        4.966536045074463,
        5.985164165496826,
        6.993622779846191,
    ]
    let predicted_s = swish(s)
    let predicted_m = swish(m)
    let predicted_l = swish(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn test_tanh(s: Tensor, m: Tensor, l: Tensor) raises -> Bool:
    let expected_s = [-0.9640275835990906, -0.7615941762924194, 0.0, 0.7615941762924194]
    let expected_m = [
        -0.9993292093276978,
        -0.9950547218322754,
        -0.9640275835990906,
        -0.7615941762924194,
        0.0,
        0.7615941762924194,
        0.9640275835990906,
        0.9950547218322754,
    ]
    let expected_l = [
        -1.0,
        -0.9999983310699463,
        -0.9999878406524658,
        -0.9999091625213623,
        -0.9993292093276978,
        -0.9950547218322754,
        -0.9640275835990906,
        -0.7615941762924194,
        0.0,
        0.7615941762924194,
        0.9640275835990906,
        0.9950547218322754,
        0.9993292093276978,
        0.9999091625213623,
        0.9999878406524658,
        0.9999983310699463,
    ]
    let predicted_s = tanh(s)
    let predicted_m = tanh(m)
    let predicted_l = tanh(l)
    return passed(
        expected_s, expected_m, expected_l, predicted_s, predicted_m, predicted_l
    )


fn print_bars():
    print("--------------------------------------------------")


fn main() raises:
    var vals = DynamicVector[Float32]()
    var shape = DynamicVector[Int]()
    shape.append(2)
    shape.append(2)

    for i in range(-2, 2):
        vals.append(i)

    let testTensor1 = Tensor(shape)._custom_fill(vals)

    shape.append(2)
    vals.clear()
    for i in range(-4, 4):
        vals.append(i)

    let testTensor2 = Tensor(shape)._custom_fill(vals)

    shape.append(2)
    vals.clear()
    for i in range(-8, 8):
        vals.append(i)

    let testTensor3 = Tensor(shape)._custom_fill(vals)

    print("Testing ReLU")
    print_bars()
    if test_relu(testTensor1, testTensor2, testTensor3):
        print("ReLU test passed")
    else:
        print("ReLU test failed")

    print_bars()
    print("Testing SELU")
    print_bars()
    if test_selu(testTensor1, testTensor2, testTensor3):
        print("SELU test passed")
    else:
        print("SELU test failed")

    print_bars()
    print("Testing Sigmoid")
    print_bars()
    if test_sigmoid(testTensor1, testTensor2, testTensor3):
        print("Sigmoid test passed")
    else:
        print("Sigmoid test failed")

    print_bars()
    print("Testing Softmax")
    print_bars()
    if test_softmax(testTensor1, testTensor2, testTensor3):
        print("Softmax test passed")
    else:
        print("Softmax test failed")

    print_bars()
    print("Testing Softplus")
    print_bars()
    if test_softplus(testTensor1, testTensor2, testTensor3):
        print("Softplus test passed")
    else:
        print("Softplus test failed")

    print_bars()
    print("Testing Softsign")
    print_bars()
    if test_softsign(testTensor1, testTensor2, testTensor3):
        print("Softsign test passed")
    else:
        print("Softsign test failed")

    print_bars()
    print("Testing Swish")
    print_bars()
    if test_swish(testTensor1, testTensor2, testTensor3):
        print("Swish test passed")
    else:
        print("Swish test failed")

    print_bars()
    print("Testing Tanh")
    print_bars()
    if test_tanh(testTensor1, testTensor2, testTensor3):
        print("Tanh test passed")
    else:
        print("Tanh test failed")
