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



    ////////////////
    int batchInputShapeGood[8] = {2, 128, 784, 128, 1, 0, 1, 99};
    int batchOutputShapeGood[8] = {2, 128, 1024, 128, 1, 0, 1, 99};
    int paramsShapeGood[8] = {2, 784, 1024, 1, 1024, 0, 1, 102};
    int biasShapeGood[8] = {2, 1, 1024, 1, 1024, 0, 1, 102};


    int batchInputShapeBad[8] = {2, 128, 781, 128, 1, 0, 1, 99};
    int batchOutputShapeBad[8] = {2, 32, 1024, 128, 1, 0, 1, 99};
    int paramsShapeBad[8] = {2, 783, 1024, 1, 1024, 0, 1, 102};
    int biasShapeBad[8] = {2, 1, 1025, 1, 1025, 0, 1, 102};

};

TEST_F(DenseLayerInputTest, InputValidationTest1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *input = new float[1000];


    int result = layer->setInput(input, inputShape3D, nullptr, nullptr);

    ASSERT_EQ(ND4J_STATUS_BAD_RANK, result);

    delete layer;
    delete[] input;
}

TEST_F(DenseLayerInputTest, InputValidationTest2) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *input = new float[100];

    int result = layer->setInput(input, inputShape2D, nullptr, nullptr);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete layer;
    delete[] input;
}

TEST_F(DenseLayerInputTest, OutputValidationTest1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[100];

    int result = layer->setOutput(output, outputShape2D);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    delete layer;
    delete[] output;
}


TEST_F(DenseLayerInputTest, OutputValidationTest2) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[1000];

    int result = layer->setOutput(output, outputShape3D);

    ASSERT_EQ(ND4J_STATUS_BAD_RANK, result);

    delete layer;
    delete[] output;
}


TEST_F(DenseLayerInputTest, JointConfiguration1) {

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *output = new float[100];

    int result = layer->configureLayer(output, inputShape2D, output, outputShape2D, 0.5f, 0.1f, nullptr);

    ASSERT_EQ(ND4J_STATUS_BAD_RNG, result);

    result = layer->configureLayer(output, inputShape2D, output, outputShape2D, 0.5f, 0.1f, output);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    ASSERT_TRUE(layer->dropOut);

    ASSERT_TRUE(layer->dropConnect);

    ASSERT_EQ(0.5f, layer->pDropOut);

    ASSERT_EQ(0.1f, layer->pDropConnect);

    delete layer;
    delete[] output;
}

TEST_F(DenseLayerInputTest, ParamsTest1) {
    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *stub = new float[10];


    int result = layer->setParameters(stub, paramsShapeGood, stub, biasShapeGood);

    ASSERT_EQ(ND4J_STATUS_OK,result);

    result = layer->configureLayer(stub, batchInputShapeGood, stub, batchOutputShapeGood, 0.0f, 0.0f, nullptr);

    ASSERT_EQ(ND4J_STATUS_OK, result);
}

TEST_F(DenseLayerInputTest, ParamsTest2) {
    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *stub = new float[10];


    int result = layer->setParameters(stub, paramsShapeBad, stub, biasShapeBad);

    ASSERT_EQ(ND4J_STATUS_BAD_SHAPE, result);

    result = layer->configureLayer(stub, inputShape2D, stub, outputShape2D, 0.0f, 0.0f, nullptr);

    ASSERT_EQ(ND4J_STATUS_BAD_INPUT, result);
}

TEST_F(DenseLayerInputTest, ParamsTest3) {
    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    float *stub = new float[10];

    int result = layer->setParameters(stub, paramsShapeGood, stub, biasShapeGood);

    ASSERT_EQ(ND4J_STATUS_OK,result);

    result = layer->configureLayer(stub, batchInputShapeBad, stub, batchOutputShapeGood, 0.0f, 0.0f, nullptr);

    ASSERT_EQ(ND4J_STATUS_BAD_INPUT, result);

    result = layer->configureLayer(stub, batchInputShapeGood, stub, batchOutputShapeBad, 0.0f, 0.0f, nullptr);

    ASSERT_EQ(ND4J_STATUS_BAD_OUTPUT, result);
}


TEST_F(DenseLayerInputTest, SGemmTest1) {
    auto *arrayA = new NDArray<float>(3, 5, 'c');
    auto *arrayB = new NDArray<float>(5, 3, 'f');
    auto *arrayC = new NDArray<float>(3, 3, 'f');

    float exp[9] = {0.0f, 60.0f, 120.f, 0.0f, 60.0f, 120.f, 0.0f, 60.0f, 120.f};

    for (int i = 0; i < arrayA->rows(); i++) {
        for (int k = 0; k < arrayA->columns(); k++) {
            arrayA->putScalar(i, k, (float) i);
        }
    }

    printf("arrayA: [");
    for (int i = 0; i < arrayA->lengthOf(); i++) {
        printf("%f, ", arrayA->getScalar(i));
    }
    printf("]\n");

    for (int i = 0; i < arrayB->rows(); i++) {
        for (int k = 0; k < arrayB->columns(); k++) {
            arrayB->putScalar(i, k, (float) (10.0 + i));
        }
    }

    printf("arrayB: [");
    for (int i = 0; i < arrayB->lengthOf(); i++) {
        printf("%f, ", arrayB->getScalar(i));
    }
    printf("]\n");

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    layer->gemmHelper(arrayA, arrayB, arrayC, 1.0f, 0.0f);

    for (int i = 0; i < arrayC->lengthOf(); i++) {
        printf("%f\n", arrayC->getScalar(i));
        ASSERT_EQ(exp[i], arrayC->getScalar(i));
    }
}

TEST_F(DenseLayerInputTest, SGemmTest2) {
    auto *arrayA = new NDArray<float>(3, 5, 'f');
    auto *arrayB = new NDArray<float>(5, 3, 'f');
    auto *arrayC = new NDArray<float>(3, 3, 'f');

    float exp[9] = {0.0f, 60.0f, 120.f, 0.0f, 60.0f, 120.f, 0.0f, 60.0f, 120.f};

    for (int i = 0; i < arrayA->rows(); i++) {
        for (int k = 0; k < arrayA->columns(); k++) {
            arrayA->putScalar(i, k, (float) i);
        }
    }

    printf("arrayA: [");
    for (int i = 0; i < arrayA->lengthOf(); i++) {
        printf("%f, ", arrayA->getScalar(i));
    }
    printf("]\n");

    for (int i = 0; i < arrayB->rows(); i++) {
        for (int k = 0; k < arrayB->columns(); k++) {
            arrayB->putScalar(i, k, (float) (10.0 + i));
        }
    }

    printf("arrayB: [");
    for (int i = 0; i < arrayB->lengthOf(); i++) {
        printf("%f, ", arrayB->getScalar(i));
    }
    printf("]\n");

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    layer->gemmHelper(arrayA, arrayB, arrayC, 1.0f, 0.0f);

    for (int i = 0; i < arrayC->lengthOf(); i++) {
        printf("%f\n", arrayC->getScalar(i));
        ASSERT_EQ(exp[i], arrayC->getScalar(i));
    }
}

TEST_F(DenseLayerInputTest, SGemmTest3) {
    auto *arrayA = new NDArray<float>(3, 5, 'f');
    auto *arrayB = new NDArray<float>(5, 3, 'f');
    auto *arrayC = new NDArray<float>(3, 3, 'c');

    float exp[9] = {0.0f, 0.0f, 0.f, 60.0f, 60.0f, 60.f, 120.0f, 120.0f, 120.f};

    for (int i = 0; i < arrayA->rows(); i++) {
        for (int k = 0; k < arrayA->columns(); k++) {
            arrayA->putScalar(i, k, (float) i);
        }
    }

    printf("arrayA: [");
    for (int i = 0; i < arrayA->lengthOf(); i++) {
        printf("%f, ", arrayA->getScalar(i));
    }
    printf("]\n");

    for (int i = 0; i < arrayB->rows(); i++) {
        for (int k = 0; k < arrayB->columns(); k++) {
            arrayB->putScalar(i, k, (float) (10.0 + i));
        }
    }

    printf("arrayB: [");
    for (int i = 0; i < arrayB->lengthOf(); i++) {
        printf("%f, ", arrayB->getScalar(i));
    }
    printf("]\n");

    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();

    layer->gemmHelper(arrayA, arrayB, arrayC, 1.0f, 0.0f);

    for (int i = 0; i < arrayC->lengthOf(); i++) {
        printf("%f\n", arrayC->getScalar(i));
        ASSERT_EQ(exp[i], arrayC->getScalar(i));
    }
}

TEST_F(DenseLayerInputTest, DropOutTest1) {
    Nd4jIndex *buffer = new Nd4jIndex[100000];

    NativeOps nativeOps;

    nd4j::random::RandomBuffer *rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);


    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();
    layer->rng = rng;
    layer->pDropOut = 0.5f;

    auto *input = new NDArray<float>(5, 5, 'c');
    input->assign(13.0f);

    auto *exp = input->dup('c');

    // input should be modified inplace here
    layer->dropOutHelper(input->buffer, input->shapeInfo);

    printf("Original array: ");
    for (int i = 0; i < exp->lengthOf(); i++) {
        printf("%f, ", exp->getScalar(i));
    }
    printf("\n");

    printf("Modified array: ");
    for (int i = 0; i < input->lengthOf(); i++) {
        printf("%f, ", input->getScalar(i));
    }
    printf("\n");

    ASSERT_FALSE(input->equalsTo(exp));

    // for dropout inverted all values here should be either 0 (dropped out) or 26.0 (scaled by 2)
    for (int i = 0; i < input->lengthOf(); i++) {
        ASSERT_TRUE(input->getScalar(i) == 0.0f || input->getScalar(i) == 26.0f);
    }
}

TEST_F(DenseLayerInputTest, DropConnectTest1) {
    Nd4jIndex *buffer = new Nd4jIndex[100000];

    NativeOps nativeOps;

    nd4j::random::RandomBuffer *rng = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, 123, 100000, (Nd4jPointer) buffer);


    nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>> *layer = new nd4j::layers::DenseLayer<float, nd4j::activations::Identity<float>>();
    layer->rng = rng;
    layer->pDropConnect = 0.5f;

    auto *input = new NDArray<float>(5, 5, 'c');
    input->assign(13.0f);

    auto *exp = input->dup('c');

    // input should be modified inplace here
    layer->dropConnectHelper(input->buffer, input->shapeInfo);

    printf("Original array: ");
    for (int i = 0; i < exp->lengthOf(); i++) {
        printf("%f, ", exp->getScalar(i));
    }
    printf("\n");

    printf("Modified array: ");
    for (int i = 0; i < input->lengthOf(); i++) {
        printf("%f, ", input->getScalar(i));
    }
    printf("\n");

    ASSERT_FALSE(input->equalsTo(exp));

    // for dropout inverted all values here should be either 0 (dropped out) or 13.0 (retained value)
    for (int i = 0; i < input->lengthOf(); i++) {
        ASSERT_TRUE(input->getScalar(i) == 0.0f || input->getScalar(i) == 13.0f);
    }
}


TEST_F(DenseLayerInputTest, FeedForwardTest1) {

    auto *weights = new NDArray<double>(784, 1000, 'f');
    auto *bias = new NDArray<double>(1, 1000, 'f');

    weights->assign(0.15f);
    bias->assign(0.13f);

    // we're checking for assign validity
    ASSERT_NEAR(0.13f, bias->meanNumber(), 1e-5);
    ASSERT_NEAR(0.15f, weights->meanNumber(), 1e-5);

    auto *input = new NDArray<double>(16, 784, 'f');
    auto *output = new NDArray<double>(16, 1000, 'f');

    input->assign(0.19f);


    auto *exp = new NDArray<double>(16, 1000, 'f');
    exp->assign(22.474001);

    auto *layer = new nd4j::layers::DenseLayer<double, nd4j::activations::Identity<double>>();

    int result = layer->setParameters(weights->buffer, weights->shapeInfo, bias->buffer, bias->shapeInfo);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    result = layer->configureLayer(input->buffer, input->shapeInfo, output->buffer, output->shapeInfo, 0.0f, 0.0f, nullptr);

    ASSERT_EQ(ND4J_STATUS_OK, result);

    result = layer->feedForward();

    ASSERT_EQ(ND4J_STATUS_OK, result);

    printf("Modified array: ");
    for (int i = 0; i < 30; i++) {
        printf("%f, ", output->getScalar(i));
    }
    printf("\n");


    printf("Exp array: ");
    for (int i = 0; i < 30; i++) {
        printf("%f, ", exp->getScalar(i));
    }
    printf("\n");


    auto meanNumber = output->meanNumber();

    printf("Output mean: %f\n", meanNumber);

    ASSERT_TRUE(exp->equalsTo(output));
}