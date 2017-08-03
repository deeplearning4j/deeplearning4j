//
// Created by raver119 on 03.08.17.
//

#include "testlayers.h"
#include <layers/generic/DenseLayer.h>

class DenseLayerInputTest : public testing::Test {
public:
    int alpha = 0;
    int inputShape3D[10] = {3, 10, 10, 10, 100, 10, 1, 0, 1, 99};
    int inputShape2D[8] = {2, 10, 10, 10, 1, 0, 1, 99};

    int outputShape2D[8] = {2, 10, 10, 10, 1, 0, 1, 99};
    int outputShape3D[10] = {3, 10, 10, 10, 100, 10, 1, 0, 1, 99};

};

TEST_F(DenseLayerInputTest, InputValidationTest1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *input = new float[1000];


    bool result = layer->setInput(input, inputShape3D, nullptr, nullptr);

    ASSERT_FALSE(result);
}

TEST_F(DenseLayerInputTest, InputValidationTest2) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *input = new float[100];

    bool result = layer->setInput(input, inputShape2D, nullptr, nullptr);

    ASSERT_TRUE(result);
}

TEST_F(DenseLayerInputTest, OutputValidationTest1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[100];

    bool result = layer->setOutput(output, outputShape2D);

    ASSERT_TRUE(result);
}


TEST_F(DenseLayerInputTest, OutputValidationTest2) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[1000];

    bool result = layer->setOutput(output, outputShape3D);

    ASSERT_FALSE(result);
}


TEST_F(DenseLayerInputTest, JointConfiguration1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[100];

    bool result = layer->configureLayer(output, inputShape2D, output, outputShape2D, 0.5f, 0.1f);

    ASSERT_TRUE(result);

    ASSERT_TRUE(layer->dropOut);

    ASSERT_TRUE(layer->dropConnect);

    ASSERT_EQ(0.5f, layer->pDropOut);

    ASSERT_EQ(0.1f, layer->pDropConnect);
}