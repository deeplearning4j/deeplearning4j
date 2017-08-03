//
// Created by raver119 on 03.08.17.
//

#include "testlayers.h"
#include <layers/generic/DenseLayer.h>

class DenseLayerInputTest : public testing::Test {
public:
    int alpha = 0;
};

TEST_F(DenseLayerInputTest, InputValidationTest) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *input = new float[1000];
    int inputShape[] = {3, 10, 10, 10, 100, 10, 1, 0, 1, 99};

    bool result = layer->setInput(input, inputShape, nullptr, nullptr);

    ASSERT_FALSE(result);
}
