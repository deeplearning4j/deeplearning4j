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



    ASSERT_TRUE(false);
}
