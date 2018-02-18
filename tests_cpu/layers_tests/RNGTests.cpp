//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <chrono>
#include <NDArray.h>
#include <NDArrayFactory.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;

class RNGTests : public testing::Test {
private:
    NativeOps nativeOps;
    Nd4jIndex *_bufferA;
    Nd4jIndex *_bufferB;

public:
    long _seed = 119L;
    nd4j::random::RandomBuffer *_rngA;
    nd4j::random::RandomBuffer *_rngB;

    NDArray<float>* nexp0 = new NDArray<float>('c', {10, 10});
    NDArray<float>* nexp1 = new NDArray<float>('c', {10, 10});
    NDArray<float>* nexp2 = new NDArray<float>('c', {10, 10});

    RNGTests() {
        _bufferA = new Nd4jIndex[100000];
        _bufferB = new Nd4jIndex[100000];
        _rngA = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);
        _rngB = (nd4j::random::RandomBuffer *) nativeOps.initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferB);

        nexp0->assign(-1.0f);
        nexp1->assign(-2.0f);
        nexp2->assign(-3.0f);
    }

    ~RNGTests() {
        nativeOps.destroyRandom(_rngA);
        nativeOps.destroyRandom(_rngB);
        delete[] _bufferA;
        delete[] _bufferB;

        delete nexp0;
        delete nexp1;
        delete nexp2;
    }
};

TEST_F(RNGTests, Test_Dropout_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOut<float>>(_rngA, nullptr, &x0, prob);
    x1.template applyRandom<randomOps::DropOut<float>>(_rngB, nullptr, &x1, prob);

    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_DropoutInverted_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    NDArrayFactory<float>::linspace(1, x0);
    NDArrayFactory<float>::linspace(1, x1);

    float prob[] = {0.5f};

    x0.template applyRandom<randomOps::DropOutInverted<float>>(_rngA, nullptr, &x0, prob);
    x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);

    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::applyDropOut(_rngA, &x0, 0.5f);
    RandomLauncher<float>::applyDropOut(_rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_2) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::applyInvertedDropOut(_rngA, &x0, 0.5f);
    RandomLauncher<float>::applyInvertedDropOut(_rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_3) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::applyAlphaDropOut(_rngA, &x0, 0.5f, 0.2f, 0.1f, 0.3f);
    RandomLauncher<float>::applyAlphaDropOut(_rngB, &x1, 0.5f, 0.2f, 0.1f, 0.3f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Uniform_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillUniform(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher<float>::fillUniform(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));

    for (int e = 0; e < x0.lengthOf(); e++) {
        float v = x0.getScalar(e);
        ASSERT_TRUE(v >= 1.0f && v <= 2.0f);
    }
}

TEST_F(RNGTests, Test_Bernoulli_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillBernoulli(_rngA, &x0, 1.0f);
    RandomLauncher<float>::fillBernoulli(_rngB, &x1, 1.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Gaussian_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillGaussian(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher<float>::fillGaussian(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_LogNormal_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillLogNormal(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher<float>::fillLogNormal(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Truncated_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillTruncatedNormal(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher<float>::fillTruncatedNormal(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Binomial_1) {
    NDArray<float> x0('c', {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillBinomial(_rngA, &x0, 3, 2.0f);
    RandomLauncher<float>::fillBinomial(_rngB, &x1, 3, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    //nexp2->printIndexedBuffer("nexp2");
    //x0.printIndexedBuffer("x0");

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Uniform_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillUniform(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(0);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_Gaussian_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillGaussian(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(6);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_LogNorm_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillLogNormal(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(10);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_TruncatedNorm_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillTruncatedNormal(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(11);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Binomial_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillBinomial(_rngB, &x1, 3, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(9);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Bernoulli_2) {
    NDArray<float> input('c', {1, 2}, {10, 10});
    NDArray<float> x1('c', {10, 10});

    RandomLauncher<float>::fillBernoulli(_rngB, &x1, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp<float>(7);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_GaussianDistribution_1) {
    NDArray<float> x('c', {2}, {10, 10});
    NDArray<float> exp0('c', {10, 10});


    nd4j::ops::random_normal<float> op;
    auto result = op.execute({&x}, {0.0, 1.0f}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));


    ASSERT_FALSE(nexp0->equalsTo(z));
    ASSERT_FALSE(nexp1->equalsTo(z));
    ASSERT_FALSE(nexp2->equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_BernoulliDistribution_1) {
    NDArray<float> x('c', {2}, {10, 10});
    NDArray<float> exp0('c', {10, 10});


    nd4j::ops::random_bernoulli<float> op;
    auto result = op.execute({&x}, {}, {3});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));


    ASSERT_FALSE(nexp0->equalsTo(z));
    ASSERT_FALSE(nexp1->equalsTo(z));
    ASSERT_FALSE(nexp2->equalsTo(z));

    delete result;
}


TEST_F(RNGTests, Test_ExponentialDistribution_1) {
    NDArray<float> x('c', {2}, {10, 10});
    NDArray<float> exp0('c', {10, 10});


    nd4j::ops::random_exponential<float> op;
    auto result = op.execute({&x}, {0.25f}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));


    ASSERT_FALSE(nexp0->equalsTo(z));
    ASSERT_FALSE(nexp1->equalsTo(z));
    ASSERT_FALSE(nexp2->equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_ExponentialDistribution_2) {
    NDArray<float> x('c', {2}, {10, 10});
    NDArray<float> y('c', {10, 10});
    NDArray<float> exp0('c', {10, 10});

    y.assign(1.0);


    nd4j::ops::random_exponential<float> op;
    auto result = op.execute({&x, &y}, {0.25f}, {0});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));


    ASSERT_FALSE(nexp0->equalsTo(z));
    ASSERT_FALSE(nexp1->equalsTo(z));
    ASSERT_FALSE(nexp2->equalsTo(z));

    delete result;
}