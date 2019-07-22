/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author raver119@gmail.com
//

#include "testlayers.h"
#include <chrono>
#include <NDArray.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/LegacyRandomOp.h>
#include <ops/declarable/CustomOperations.h>

using namespace nd4j;

class RNGTests : public testing::Test {
private:
    //Nd4jLong *_bufferA;
    //Nd4jLong *_bufferB;

public:
    long _seed = 119L;
    //nd4j::random::RandomBuffer *_rngA;
    //nd4j::random::RandomBuffer *_rngB;
    nd4j::graph::RandomGenerator _rngA;
    nd4j::graph::RandomGenerator _rngB;

    NDArray* nexp0 = NDArrayFactory::create_<float>('c', {10, 10});
    NDArray* nexp1 = NDArrayFactory::create_<float>('c', {10, 10});
    NDArray* nexp2 = NDArrayFactory::create_<float>('c', {10, 10});

    RNGTests() {
        //_bufferA = new Nd4jLong[100000];
        //_bufferB = new Nd4jLong[100000];
        //_rngA = (nd4j::random::RandomBuffer *) initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferA);
        //_rngB = (nd4j::random::RandomBuffer *) initRandom(nullptr, _seed, 100000, (Nd4jPointer) _bufferB);
        _rngA.setStates(_seed, _seed);
        _rngB.setStates(_seed, _seed);
        nexp0->assign(-1.0f);
        nexp1->assign(-2.0f);
        nexp2->assign(-3.0f);
    }

    ~RNGTests() {
        //destroyRandom(_rngA);
        //destroyRandom(_rngB);
        //delete[] _bufferA;
        //delete[] _bufferB;

        delete nexp0;
        delete nexp1;
        delete nexp2;
    }
};

TEST_F(RNGTests, TestSeeds_1) {
    RandomGenerator generator(123L, 456L);

    ASSERT_EQ(123, generator.rootState());
    ASSERT_EQ(456, generator.nodeState());

    Nd4jPointer ptr = malloc(sizeof(RandomGenerator));
    memcpy(ptr, &generator, sizeof(RandomGenerator));

    auto cast = reinterpret_cast<RandomGenerator*>(ptr);
    ASSERT_EQ(123, cast->rootState());
    ASSERT_EQ(456, cast->nodeState());

    free(ptr);
}

TEST_F(RNGTests, TestSeeds_2) {
    RandomGenerator generator(12, 13);

    generator.setStates(123L, 456L);

    ASSERT_EQ(123, generator.rootState());
    ASSERT_EQ(456, generator.nodeState());
}


TEST_F(RNGTests, Test_Dropout_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    x0.linspace(1);
    x1.linspace(1);

    float prob[] = {0.5f};

    //x0.applyRandom(random::DropOut, _rngA, nullptr, &x0, prob);
    //x1.applyRandom(random::DropOut, _rngB, nullptr, &x1, prob);
    RandomLauncher::applyDropOut(LaunchContext::defaultContext(), _rngA, &x0, 0.5);
    RandomLauncher::applyDropOut(LaunchContext::defaultContext(), _rngB, &x1, 0.5);
    ASSERT_TRUE(x0.equalsTo(&x1));
    //x0.printIndexedBuffer("Dropout");
    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_DropoutInverted_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    x0.linspace(1);
    x1.linspace(1);

    float prob[] = {0.5f};

    //x0.template applyRandom<randomOps::DropOutInverted<float>>(_rngA, nullptr, &x0, prob);
    //x1.template applyRandom<randomOps::DropOutInverted<float>>(_rngB, nullptr, &x1, prob);
    RandomLauncher::applyInvertedDropOut(LaunchContext::defaultContext(), _rngA, &x0, 0.5);
    RandomLauncher::applyInvertedDropOut(LaunchContext::defaultContext(), _rngB, &x1, 0.5);
    ASSERT_TRUE(x0.equalsTo(&x1));
    //x0.printIndexedBuffer("DropoutInverted");
    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyDropOut(LaunchContext::defaultContext(), _rngA, &x0, 0.5f);
    RandomLauncher::applyDropOut(LaunchContext::defaultContext(), _rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_2) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyInvertedDropOut(LaunchContext::defaultContext(), _rngA, &x0, 0.5f);
    RandomLauncher::applyInvertedDropOut(LaunchContext::defaultContext(), _rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_3) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyAlphaDropOut(LaunchContext::defaultContext(), _rngA, &x0, 0.5f, 0.2f, 0.1f, 0.3f);
    RandomLauncher::applyAlphaDropOut(LaunchContext::defaultContext(), _rngB, &x1, 0.5f, 0.2f, 0.1f, 0.3f);

    //x1.printIndexedBuffer("x1");
    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Uniform_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillUniform(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));

    for (int e = 0; e < x0.lengthOf(); e++) {
        float v = x0.e<float>(e);
        ASSERT_TRUE(v >= 1.0f && v <= 2.0f);
    }
}

TEST_F(RNGTests, Test_Uniform_3) {
    auto x0 = NDArrayFactory::create<double>('c', {1000000});

    RandomLauncher::fillUniform(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);

    for (int e = 0; e < x0.lengthOf(); e++) {
        auto v = x0.t<double>(e);
        ASSERT_TRUE(v >= 1.0 && v <= 2.0);
    }
}

TEST_F(RNGTests, Test_Bernoulli_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBernoulli(LaunchContext::defaultContext(), _rngA, &x0, 1.0f);
    RandomLauncher::fillBernoulli(LaunchContext::defaultContext(), _rngB, &x1, 1.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Gaussian_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    //x0.printIndexedBuffer("x0");
    //x1.printIndexedBuffer("x1");
    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Gaussian_21) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngA, &x0, 0.0f, 1.0f);
    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngB, &x1, 0.0f, 1.0f);

    //x0.printIndexedBuffer("x0");
    //x1.printIndexedBuffer("x1");
    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
    nd4j::ops::moments op;
    auto result = op.execute({&x0}, {}, {});
    //x0.printIndexedBuffer("X0 Normal");
    //x1.printIndexedBuffer("X1 Normal");
    ASSERT_TRUE(result->status() == Status::OK());
    auto mean = result->at(0);
    auto variance = result->at(1);

    // mean->printIndexedBuffer("Mean");
    // variance->printIndexedBuffer("Variance");

    ASSERT_NEAR(nd4j::math::nd4j_abs(mean->e<float>(0)), 0.f, 0.2f);
    ASSERT_NEAR(variance->e<float>(0), 1.0f, 0.2f);

    delete result;
}

#ifndef DEBUG_BUILD
TEST_F(RNGTests, Test_Gaussian_22) {
    auto x0 = NDArrayFactory::create<float>('c', {10000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {10000, 1000});

    RandomLauncher::fillGaussian(nd4j::LaunchContext::defaultContext(), _rngA, &x0, 0.0f, 1.0f);
    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngB, &x1, 0.0f, 1.0f);

    //x0.printIndexedBuffer("x0");
    //x1.printIndexedBuffer("x1");
    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
    nd4j::ops::moments op;
    auto result = op.execute({&x0}, {}, {});
    //x0.printIndexedBuffer("X0 Normal");
    //x1.printIndexedBuffer("X1 Normal");
    ASSERT_TRUE(result->status() == Status::OK());
    auto mean0 = result->at(0);
    auto variance0 = result->at(1);

    //mean0->printIndexedBuffer("Mean");
    //variance0->printIndexedBuffer("Variance");
    ASSERT_NEAR(nd4j::math::nd4j_abs(mean0->e<float>(0)), 0.f, 1.0e-3f);
    ASSERT_NEAR(variance0->e<float>(0), 1.0f, 1.e-3f);
    delete result;
}

TEST_F(RNGTests, Test_Gaussian_3) {
    auto x0 = NDArrayFactory::create<double>('c', {10000000});

    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngA, &x0, 0.0, 1.0);

    auto mean = x0.meanNumber().e<double>(0);
    auto stdev = x0.varianceNumber(nd4j::variance::SummaryStatsStandardDeviation, false).e<double>(0);

    ASSERT_NEAR(0.0, mean, 1e-3);
    ASSERT_NEAR(1.0, stdev, 1e-3);
}

TEST_F(RNGTests, Test_LogNormal_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillLogNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillLogNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Truncated_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 1.0");
    auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 2.0");
    // x1.printIndexedBuffer("Distribution TN");

}
TEST_F(RNGTests, Test_Truncated_2) {
    auto x0 = NDArrayFactory::create<float>('c', {1000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {1000, 1000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    //ASSERT_FALSE(x0.equalsTo(nexp0));
    //ASSERT_FALSE(x0.equalsTo(nexp1));
    //ASSERT_FALSE(x0.equalsTo(nexp2));

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 1.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 2.0");
    //x1.printIndexedBuffer("Distribution TN");
    ASSERT_NEAR(mean.e<float>(0), 1.f, 0.5);
    ASSERT_NEAR(deviation.e<float>(0), 2.f, 0.5);
}

TEST_F(RNGTests, Test_Truncated_21) {
    auto x0 = NDArrayFactory::create<float>('c', {1000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {1000, 1000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    auto mean0 = x0.reduceNumber(reduce::Mean);
    // mean0.printIndexedBuffer("0Mean 1.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation0 = x0.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    // deviation0.printIndexedBuffer("0Deviation should be 2.0");

    //ASSERT_FALSE(x0.equalsTo(nexp0));
    //ASSERT_FALSE(x0.equalsTo(nexp1));
    //ASSERT_FALSE(x0.equalsTo(nexp2));

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 1.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 2.0");
    //x1.printIndexedBuffer("Distribution TN");
    ASSERT_NEAR(mean.e<float>(0), 1.f, 0.002);
    ASSERT_NEAR(deviation.e<float>(0), 2.f, 0.5);
    nd4j::ops::moments op;
    auto result = op.execute({&x0}, {}, {}, {}, false, nd4j::DataType::FLOAT32);
    // result->at(0)->printBuffer("MEAN");
    // result->at(1)->printBuffer("VARIANCE");
    delete result;
    nd4j::ops::reduce_min minOp;
    nd4j::ops::reduce_max maxOp;
    auto minRes = minOp.execute({&x1}, {}, {}, {});
    auto maxRes = maxOp.execute({&x0}, {}, {}, {});
    // minRes->at(0)->printBuffer("MIN for Truncated");
    // maxRes->at(0)->printBuffer("MAX for Truncated");

    delete minRes;
    delete maxRes;
}

TEST_F(RNGTests, Test_Truncated_22) {
    auto x0 = NDArrayFactory::create<float>('c', {1000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {1000, 1000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 2.0f, 4.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 2.0f, 4.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    auto mean0 = x0.reduceNumber(reduce::Mean);
    // mean0.printIndexedBuffer("0Mean 2.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation0 = x0.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    // deviation0.printIndexedBuffer("0Deviation should be 4.0");

    //ASSERT_FALSE(x0.equalsTo(nexp0));
    //ASSERT_FALSE(x0.equalsTo(nexp1));
    //ASSERT_FALSE(x0.equalsTo(nexp2));

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 2.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 4.0");
    //x1.printIndexedBuffer("Distribution TN");
    ASSERT_NEAR(mean.e<float>(0), 2.f, 0.01);
    ASSERT_NEAR(deviation.e<float>(0), 4.f, 0.5);
    nd4j::ops::moments op;
    auto result = op.execute({&x0}, {}, {}, {}, false, nd4j::DataType::FLOAT32);
    // result->at(0)->printBuffer("MEAN");
    // result->at(1)->printBuffer("VARIANCE");
    delete result;
    nd4j::ops::reduce_min minOp;
    nd4j::ops::reduce_max maxOp;
    auto minRes = minOp.execute({&x1}, {}, {}, {});
    auto maxRes = maxOp.execute({&x0}, {}, {}, {});
    // minRes->at(0)->printBuffer("MIN for Truncated2");
    // maxRes->at(0)->printBuffer("MAX for Truncated2");

    delete minRes;
    delete maxRes;
}

TEST_F(RNGTests, Test_Truncated_23) {
    auto x0 = NDArrayFactory::create<float>('c', {1000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {1000, 1000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 0.0f, 1.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 0.0f, 1.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    auto mean0 = x0.reduceNumber(reduce::Mean);
    // mean0.printIndexedBuffer("0Mean 2.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation0 = x0.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    // deviation0.printIndexedBuffer("0Deviation should be 4.0");

    //ASSERT_FALSE(x0.equalsTo(nexp0));
    //ASSERT_FALSE(x0.equalsTo(nexp1));
    //ASSERT_FALSE(x0.equalsTo(nexp2));

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 2.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 4.0");
    //x1.printIndexedBuffer("Distribution TN");
    ASSERT_NEAR(mean.e<float>(0), 0.f, 0.01);
    ASSERT_NEAR(deviation.e<float>(0), 1.f, 0.5);
    nd4j::ops::moments op;
    auto result = op.execute({&x0}, {}, {}, {}, false, nd4j::DataType::FLOAT32);
    // result->at(0)->printBuffer("MEAN");
    // result->at(1)->printBuffer("VARIANCE");
    delete result;
    nd4j::ops::reduce_min minOp;
    nd4j::ops::reduce_max maxOp;
    auto minRes = minOp.execute({&x1}, {}, {}, {});
    auto maxRes = maxOp.execute({&x0}, {}, {}, {});
    // minRes->at(0)->printBuffer("MIN for Truncated3");
    // maxRes->at(0)->printBuffer("MAX for Truncated3");

    delete minRes;
    delete maxRes;
}

TEST_F(RNGTests, Test_Truncated_3) {
    auto x0 = NDArrayFactory::create<float>('c', {10000, 1000});
    auto x1 = NDArrayFactory::create<float>('c', {10000, 1000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    //ASSERT_FALSE(x0.equalsTo(nexp0));
    //ASSERT_FALSE(x0.equalsTo(nexp1));
    //ASSERT_FALSE(x0.equalsTo(nexp2));

    // Check up distribution
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 1.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 2.0");
    //x1.printIndexedBuffer("Distribution TN");
    ASSERT_NEAR(mean.e<float>(0), 1.f, 0.001);
    ASSERT_NEAR(deviation.e<float>(0), 2.f, 0.3);
}
#endif

TEST_F(RNGTests, Test_Binomial_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBinomial(LaunchContext::defaultContext(), _rngA, &x0, 3, 2.0f);
    RandomLauncher::fillBinomial(LaunchContext::defaultContext(), _rngB, &x1, 3, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    //nexp2->printIndexedBuffer("nexp2");
    //x0.printIndexedBuffer("x0");

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Uniform_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillUniform(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(0);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_Gaussian_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(random::GaussianDistribution);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_LogNorm_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillLogNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(random::LogNormalDistribution);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_TruncatedNorm_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(random::TruncatedNormalDistribution);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));
    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Binomial_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBinomial(LaunchContext::defaultContext(), _rngB, &x1, 3, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp(random::BinomialDistributionEx);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {3});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Bernoulli_2) {
    auto input = NDArrayFactory::create<Nd4jLong>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBernoulli(LaunchContext::defaultContext(), _rngB, &x1, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp(random::BernoulliDistribution);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {});

    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_GaussianDistribution_1) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {10, 10});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 10});


    nd4j::ops::random_normal op;
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
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {10, 10});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 10});


    nd4j::ops::random_bernoulli op;
    auto result = op.execute({&x}, {0.5f}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_FALSE(exp0.equalsTo(z));

    ASSERT_FALSE(nexp0->equalsTo(z));
    ASSERT_FALSE(nexp1->equalsTo(z));
    ASSERT_FALSE(nexp2->equalsTo(z));

    delete result;
}


TEST_F(RNGTests, Test_ExponentialDistribution_1) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {10, 10});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 10});


    nd4j::ops::random_exponential op;
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
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2}, {10, 10});
    auto y = NDArrayFactory::create<float>('c', {10, 10});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 10});

    y.assign(1.0);


    nd4j::ops::random_exponential op;
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

namespace nd4j {
    namespace tests {
        static void fillList(Nd4jLong seed, int numberOfArrays, std::vector<Nd4jLong> &shape, std::vector<NDArray*> &list, nd4j::graph::RandomGenerator *rng) {
            rng->setSeed((int) seed);
            
            for (int i = 0; i < numberOfArrays; i++) {
                auto array = NDArrayFactory::create_<double>('c', shape);

                nd4j::ops::randomuniform op;
                op.execute(*rng, {array}, {array}, {0.0, 1.0}, {}, {}, true);

                list.emplace_back(array);
            }
        };
    }
}

TEST_F(RNGTests, Test_Reproducibility_9) { 
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, seed, bufferSize, buffer);

    const int length = 4000000;
    int *arrayE = new int[length];
    int *arrayT = new int[length];

    for (int e = 0; e < length; e++)
        arrayE[e] = rng->relativeInt(e);

    rng->rewindH(static_cast<Nd4jLong>(length));

    refreshBuffer(nullptr, seed, reinterpret_cast<Nd4jPointer>(rng));

    for (int e = 0; e < length; e++)
        arrayT[e] = rng->relativeInt(e);

    rng->rewindH(static_cast<Nd4jLong>(length));
    
    for (int e = 0; e < length; e++)
        if (arrayE[e] != arrayT[e]) {
            // nd4j_printf("Failed at index[%i]\n", e);
            ASSERT_TRUE(false);
        }

    delete[] arrayE;
    delete[] arrayT;

    destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_Reproducibility_8) { 
    Nd4jLong seed = 123;

    std::vector<int> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, seed, bufferSize, buffer);

    const int length = 4000000;
    int *arrayE = new int[length];
    int *arrayT = new int[length];

    for (int e = 0; e < length; e++)
        arrayE[e] = static_cast<int>(rng->relativeT<float>(e));

    rng->rewindH(static_cast<Nd4jLong>(length));

    refreshBuffer(nullptr, seed, reinterpret_cast<Nd4jPointer>(rng));

    for (int e = 0; e < length; e++)
        arrayT[e] = static_cast<int>(rng->relativeT<float>(e));

    rng->rewindH(static_cast<Nd4jLong>(length));
    
    for (int e = 0; e < length; e++)
        if (arrayE[e] != arrayT[e]) {
            // nd4j_printf("Failed at index[%i]\n", e);
            ASSERT_TRUE(false);
        }

    delete[] arrayE;
    delete[] arrayT;

    destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_RandomBuffer_Half_1) {
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) initRandom(nullptr, seed, bufferSize, buffer);

    auto r0 = rng->relativeT<float16>(12L);
    auto r1 = rng->relativeT<float16>(13L);

    ASSERT_NE(r0, r1);

    destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_Reproducibility_1) {
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    nd4j::graph::RandomGenerator rng;

    std::vector<NDArray*> expList;
    nd4j::tests::fillList(seed, 10, shape, expList, &rng);

    for (int e = 0; e < 2; e++) {
        std::vector<NDArray *> trialList;
        nd4j::tests::fillList(seed, 10, shape, trialList, &rng);

        for (int a = 0; a < expList.size(); a++) {
            auto arrayE = expList[a];
            auto arrayT = trialList[a];

            bool t = arrayE->equalsTo(arrayT);
            if (!t) {
                // nd4j_printf("Failed at iteration [%i] for array [%i]\n", e, a);
                ASSERT_TRUE(false);
            }

            delete arrayT;
        }
    }

    for (auto v: expList)
            delete v;
}

#ifndef DEBUG_BUILD
TEST_F(RNGTests, Test_Reproducibility_2) {
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 64, 64};
    nd4j::graph::RandomGenerator rng;

    std::vector<NDArray*> expList;
    nd4j::tests::fillList(seed, 10, shape, expList, &rng);

    for (int e = 0; e < 2; e++) {
        std::vector<NDArray*> trialList;
        nd4j::tests::fillList(seed, 10, shape, trialList, &rng);

        for (int a = 0; a < expList.size(); a++) {
            auto arrayE = expList[a];
            auto arrayT = trialList[a];

            bool t = arrayE->equalsTo(arrayT);
            if (!t) {
                // nd4j_printf("Failed at iteration [%i] for array [%i]\n", e, a);

                for (Nd4jLong f = 0; f < arrayE->lengthOf(); f++) {
                    double x = arrayE->e<double>(f);
                    double y = arrayT->e<double>(f);

                    if (nd4j::math::nd4j_re(x, y) > 0.1) {
                        // nd4j_printf("E[%lld] %f != T[%lld] %f\n", (long long) f, (float) x, (long long) f, (float) y);
                        throw std::runtime_error("boom");
                    }
                }

                // just breaker, since test failed
                ASSERT_TRUE(false);
            }

            delete arrayT;
        }
    }

    for (auto v: expList)
            delete v;
}

TEST_F(RNGTests, Test_Uniform_4) {
    auto x1 = NDArrayFactory::create<double>('c', {1000000});

    RandomLauncher::fillUniform(LaunchContext::defaultContext(), _rngB, &x1, 1.0, 2.0);

    /* Check up distribution */
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean should be 1.5");
    auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsVariance, false);
    //deviation /= (double)x1.lengthOf();
    // deviation.printIndexedBuffer("Deviation should be 1/12 (0.083333)");

    ASSERT_NEAR(mean.e<double>(0), 1.5, 1e-3);
    ASSERT_NEAR(1/12., deviation.e<double>(0), 1e-3);
}
#endif

TEST_F(RNGTests, test_choice_1) {
    auto x = NDArrayFactory::linspace<double>(0, 10, 11);
    auto prob = NDArrayFactory::valueOf<double>({11}, 1.0/11, 'c');
    auto z = NDArrayFactory::create<double>('c', {1000});

    RandomGenerator rng(119, 256);
    NativeOpExecutioner::execRandom(nd4j::LaunchContext ::defaultContext(), random::Choice, &rng, x->buffer(), x->shapeInfo(), x->specialBuffer(), x->specialShapeInfo(), prob->buffer(), prob->shapeInfo(), prob->specialBuffer(), prob->specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr);

    // z.printIndexedBuffer("z");

    delete x;
    delete prob;
}
