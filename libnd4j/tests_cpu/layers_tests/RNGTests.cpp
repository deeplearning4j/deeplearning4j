/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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

//    x0.printIndexedBuffer("x0");
//    x1.printIndexedBuffer("x1");
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

#ifdef DEBUG_BUILD
TEST_F(RNGTests, Test_Gaussian_22) {
    auto x0 = NDArrayFactory::create<float>('c', {1000, 800});
    auto x1 = NDArrayFactory::create<float>('c', {1000, 800});

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
    auto x0 = NDArrayFactory::create<double>('c', {800000});

    RandomLauncher::fillGaussian(LaunchContext::defaultContext(), _rngA, &x0, 0.0, 1.0);

    auto mean = x0.meanNumber(); //.e<double>(0);
    auto stdev = x0.varianceNumber(nd4j::variance::SummaryStatsStandardDeviation, false);//.e<double>(0);
    auto meanExp = NDArrayFactory::create<double>(0.);
    auto devExp = NDArrayFactory::create<double>(1.);
    ASSERT_TRUE(meanExp.equalsTo(mean, 1.e-3));
    ASSERT_TRUE(devExp.equalsTo(stdev, 1.e-3));
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
    auto x0 = NDArrayFactory::create<float>('c', {100, 100});
    auto x1 = NDArrayFactory::create<float>('c', {100, 100});

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
    auto x0 = NDArrayFactory::create<float>('c', {100, 100});
    auto x1 = NDArrayFactory::create<float>('c', {100, 100});

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
    ASSERT_NEAR(deviation.e<float>(0), 4.f, 0.52);
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
    auto x0 = NDArrayFactory::create<float>('c', {2000, 2000});
    auto x1 = NDArrayFactory::create<float>('c', {2000, 2000});

    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(LaunchContext::defaultContext(), _rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    // Check up distribution
    auto mean = x1.reduceNumber(reduce::Mean);
    // mean.printIndexedBuffer("Mean 1.0");
    //auto sumA = x1 - mean; //.reduceNumber(reduce::Sum);

    auto deviation = x1.varianceNumber(variance::SummaryStatsStandardDeviation, false);
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

TEST_F(RNGTests, Test_PoissonDistribution_1) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {1}, {10});
    auto la = NDArrayFactory::create<float>('c', {2, 3});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 2, 3});

    la.linspace(1.0);


    nd4j::ops::random_poisson op;
    auto result = op.execute({&x, &la}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Poisson distribution");
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_GammaDistribution_1) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {1}, {10});
    auto al = NDArrayFactory::create<float>('c', {2, 3});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 2, 3});

    al.linspace(1.0);


    nd4j::ops::random_gamma op;
    auto result = op.execute({&x, &al}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Gamma distribution");
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_GammaDistribution_2) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {1}, {10});
    auto al = NDArrayFactory::create<float>('c', {2, 3});
    auto be = NDArrayFactory::create<float>('c', {2, 3});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 2, 3});

    al.linspace(1.0);
    be.assign(1.0);

    nd4j::ops::random_gamma op;
    auto result = op.execute({&x, &al, &be}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Gamma distribution");
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_GammaDistribution_3) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {1}, {10});
    auto al = NDArrayFactory::create<float>('c', {3, 1});
    auto be = NDArrayFactory::create<float>('c', {1, 2});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 3, 2});

    al.linspace(1.0);
    be.assign(2.0);

    nd4j::ops::random_gamma op;
    auto result = op.execute({&x, &al, &be}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
//    z->printIndexedBuffer("Gamma distribution");
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));

    delete result;
}

TEST_F(RNGTests, Test_UniformDistribution_04) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {1}, {10});
    auto al = NDArrayFactory::create<int>(1);
    auto be = NDArrayFactory::create<int>(20);
    auto exp0 = NDArrayFactory::create<float>('c', {10});


    nd4j::ops::randomuniform op;
    auto result = op.execute({&x, &al, &be}, {}, {DataType::INT32});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(exp0.isSameShape(z));
    ASSERT_FALSE(exp0.equalsTo(z));

    delete result;
}

namespace nd4j {
    namespace tests {
        static void fillList(Nd4jLong seed, int numberOfArrays, std::vector<Nd4jLong> &shape, std::vector<NDArray*> &list, nd4j::graph::RandomGenerator *rng) {
            rng->setSeed((int) seed);

            for (int i = 0; i < numberOfArrays; i++) {
                auto arrayI = NDArrayFactory::create<Nd4jLong>(shape);
                auto arrayR = NDArrayFactory::create_<double>('c', shape);
                auto min = NDArrayFactory::create(0.0);
                auto max = NDArrayFactory::create(1.0);
                nd4j::ops::randomuniform op;
                op.execute(*rng, {&arrayI, &min, &max}, {arrayR}, {}, {DataType::DOUBLE}, {}, false);

                list.emplace_back(arrayR);
            }
        };
    }
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

TEST_F(RNGTests, test_uniform_119) {
    auto x = NDArrayFactory::create<int>('c', {2}, {1, 5});
    auto z = NDArrayFactory::create<float>('c', {1, 5});


    nd4j::ops::randomuniform op;
    auto status = op.execute({&x}, {&z}, {1.0, 2.0}, {}, {});
    ASSERT_EQ(Status::OK(), status);
}

TEST_F(RNGTests, test_multinomial_1) {
    
    NDArray probs('f', { 3, 3 }, { 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3 }, nd4j::DataType::FLOAT32);
    NDArray expected('f', { 3, 3 }, { 0, 1, 2,  2, 0, 0,  1, 2, 1 }, nd4j::DataType::INT64);
    NDArray output('f', { 3, 3 }, nd4j::DataType::INT64);
    NDArray samples('f', { 1 }, { 3 }, nd4j::DataType::INT32);
    
    nd4j::ops::random_multinomial op;
    RandomGenerator rng(1234, 1234);
    ASSERT_EQ(Status::OK(),  op.execute(rng, { &probs, &samples }, { &output }, {}, { 0, INT64}, {}, false) );
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
   
    NDArray probsZ('c', { 1, 3 }, { 0.3, 0.3, 0.3 }, nd4j::DataType::FLOAT32);
    NDArray expectedZ('c', { 3, 3 }, { 0, 0, 0,  0, 0, 0,  0, 0, 0 }, nd4j::DataType::INT64);

    auto result = op.execute({ &probsZ, &samples }, { }, { 1, INT64 });
    auto outputZ = result->at(0);
    
    ASSERT_EQ(Status::OK(), result->status());
    ASSERT_TRUE(expectedZ.isSameShape(outputZ));
    ASSERT_TRUE(expectedZ.equalsTo(outputZ));
    delete result;
}

TEST_F(RNGTests, test_multinomial_2) {

    NDArray samples('c', { 1 }, { 20 }, nd4j::DataType::INT32);
    NDArray probs('c', { 3, 5 }, { 0.2, 0.3, 0.5,    0.3, 0.5, 0.2,  0.5, 0.2, 0.3,  0.35, 0.25, 0.3,  0.25, 0.25, 0.5 }, nd4j::DataType::FLOAT32);
    NDArray expected('c', { 3, 20 }, { 0, 2, 0, 2, 0, 4, 2, 0, 1, 2, 0, 2, 3, 0, 0, 2, 4, 4, 1, 0, 2, 3, 2, 3, 0, 1, 3, 1, 1, 1, 2, 4, 3, 3, 1, 4, 4, 2, 0, 0, 3, 3, 3, 0, 0, 2, 2, 3, 3, 0,  0, 2, 3, 4, 2, 2, 3, 2, 1, 2   }, nd4j::DataType::INT64);
    NDArray output('c', { 3, 20 }, nd4j::DataType::INT64);

    nd4j::ops::random_multinomial op;
    RandomGenerator rng(1234, 1234);
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &output }, {}, { 0, INT64 }, {}, false));
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
    
    NDArray probs2('c', { 5, 3 }, { 0.2, 0.3, 0.5,    0.3, 0.5, 0.2,  0.5, 0.2, 0.3,  0.35, 0.25, 0.3,  0.25, 0.25, 0.5 }, nd4j::DataType::FLOAT32);
    NDArray expected2('c', { 20, 3 }, {  0, 2, 3, 2, 3, 3, 0, 2, 3, 2,  3, 0, 0, 0, 0, 4, 1, 2, 2, 3,  2, 3, 1, 3, 1, 1, 3, 2, 1, 0, 0, 2, 0, 2, 4, 2, 3, 3, 3, 0,  3, 4, 0, 1, 2, 2, 0, 2, 4, 4, 0, 4, 2, 2, 1, 0, 1, 0, 0, 2  }, nd4j::DataType::INT64);
    NDArray output2('c', { 20, 3 }, nd4j::DataType::INT64);
    
    rng.setStates(1234, 1234);
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs2, &samples }, { &output2 }, {}, { 1, INT64 }, {}, false));
    ASSERT_TRUE(expected2.isSameShape(output2));
    ASSERT_TRUE(expected2.equalsTo(output2));
}

TEST_F(RNGTests, test_multinomial_3) {
    
    NDArray probs('c', {  4, 3 }, { 0.3, 0.3, 0.4,  0.3, 0.4, 0.3,  0.3, 0.3, 0.4,  0.4, 0.3, 0.3 }, nd4j::DataType::FLOAT32);
    NDArray  expected('c', { 4, 5 }, nd4j::DataType::INT64);
    NDArray  output('c', { 4, 5 }, nd4j::DataType::INT64);
    NDArray samples('c', { 1 }, { 5 }, nd4j::DataType::INT32);
    RandomGenerator rng(1234, 1234);

    nd4j::ops::random_multinomial op;
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &expected }, {}, { 0, INT64 }, {}, false));
   
    rng.setStates(1234, 1234);
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &output }, {}, { 0, INT64 }, {}, false));
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

TEST_F(RNGTests, test_multinomial_4) {

    NDArray probs('c', { 3, 4 }, { 0.3, 0.3, 0.4, 0.3, 0.4, 0.3, 0.3, 0.3, 0.4, 0.4, 0.3, 0.3 }, nd4j::DataType::FLOAT32);
    NDArray  expected('c', { 5, 4 }, nd4j::DataType::INT64);
    NDArray  output('c', { 5, 4 }, nd4j::DataType::INT64);
    NDArray samples('c', { 1 }, { 5 }, nd4j::DataType::INT32);

    RandomGenerator rng(1234, 1234);
    nd4j::ops::random_multinomial op;
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &expected }, {}, { 1, INT64 }, {}, false));

    rng.setStates(1234, 1234);
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &output }, {}, { 1, INT64 }, {}, false));
    ASSERT_TRUE(expected.isSameShape(output));
    ASSERT_TRUE(expected.equalsTo(output));
}

TEST_F(RNGTests, test_multinomial_5) {
    // multinomial as binomial if 2 classes used
    int batchValue = 1;
    int ClassValue = 2;
    int Samples = 1000000;

    NDArray samples('c', { 1 }, { 1.*Samples }, nd4j::DataType::INT32);
    
    NDArray probs('c', { ClassValue, batchValue }, { 1.0, 1.0 }, nd4j::DataType::FLOAT32);
    
    nd4j::ops::random_multinomial op;

    NDArray  output('c', { Samples, batchValue }, nd4j::DataType::INT64);
    RandomGenerator rng(1234, 1234);
    
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &output }, {}, { 1 }, {}, false));
    
    auto deviation = output.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    auto mean = output.meanNumber();
    // printf("Var: %f  Mean: %f \n", deviation.e<double>(0), mean.e<double>(0));
    // theoretical values for binomial
    ASSERT_NEAR(0.5, deviation.e<double>(0), 3e-3);
    ASSERT_NEAR(0.5, mean.e<double>(0), 3e-3);
    
    for (int i = 0; i < output.lengthOf(); i++) {
        auto value = output.e<Nd4jLong>(i);
        ASSERT_TRUE(value >= 0 && value < ClassValue);
    }

    auto resultR = op.execute({ &probs, &samples }, { }, { 1 });
    auto outputR = resultR->at(0);
    ASSERT_EQ(Status::OK(), resultR->status());

    deviation = outputR->varianceNumber(variance::SummaryStatsStandardDeviation, false);
    mean = outputR->meanNumber();
    // printf("Random seed - Var: %f  Mean: %f \n", deviation.e<double>(0), mean.e<double>(0));
    ASSERT_NEAR(0.5, deviation.e<double>(0), 35e-3);
    ASSERT_NEAR(0.5, mean.e<double>(0), 35e-3);

    for (int i = 0; i < outputR->lengthOf(); i++) {
        auto value = outputR->e<Nd4jLong>(i);
        ASSERT_TRUE(value >= 0 && value < ClassValue);
    }

    delete resultR;
}


TEST_F(RNGTests, test_multinomial_6) {

    int batchValue = 1;
    int ClassValue = 5;
    int Samples = 1000000;
    
    NDArray samples('c', { 1 }, { 1. * Samples }, nd4j::DataType::INT32);

    nd4j::ops::random_multinomial op;
    NDArray probExpect('c', { ClassValue }, { 0.058, 0.096, 0.1576, 0.2598, 0.4287 }, nd4j::DataType::DOUBLE);

    // without seed
    NDArray probsR('c', { batchValue,  ClassValue }, { 1., 1.5, 2., 2.5, 3. }, nd4j::DataType::FLOAT32);

    auto resultR = op.execute({ &probsR, &samples }, { }, { 0 });
    auto outputR = resultR->at(0);
    ASSERT_EQ(Status::OK(), resultR->status());

    NDArray countsR('c', { ClassValue }, { 0, 0, 0, 0, 0 }, nd4j::DataType::DOUBLE);
    
    for (int i = 0; i < outputR->lengthOf(); i++) {
        auto value = outputR->e<Nd4jLong>(i);
        ASSERT_TRUE(value >= 0 && value < ClassValue);
        double* z = countsR.bufferAsT<double>();
        z[value] += 1;
    }

    for (int i = 0; i < countsR.lengthOf(); i++) {
        auto c = countsR.e<double>(i);
        auto p = probExpect.e<double>(i);
        // printf("Get freq : %f  Expect freq: %f \n", c / Samples, p);
        ASSERT_NEAR((c / Samples), p, 35e-3);
    }

    auto deviation = outputR->varianceNumber(variance::SummaryStatsStandardDeviation, false);
    auto mean = outputR->meanNumber();
    // printf("Var: %f  Mean: %f \n", deviation.e<double>(0), mean.e<double>(0));
    ASSERT_NEAR(1.2175, deviation.e<double>(0), 35e-3);
    ASSERT_NEAR(2.906, mean.e<double>(0), 35e-3);

    delete resultR;

    RandomGenerator rng(1234, 1234);
    NDArray probs('c', { batchValue, ClassValue }, { 1., 1.5, 2., 2.5, 3. }, nd4j::DataType::FLOAT32);
    NDArray  output('c', { batchValue, Samples }, nd4j::DataType::INT64);
    
    ASSERT_EQ(Status::OK(), op.execute(rng, { &probs, &samples }, { &output }, {}, { 0, INT64 }, {}, false));

    NDArray counts('c', { ClassValue }, { 0, 0, 0, 0, 0 }, nd4j::DataType::DOUBLE);
    
    for (int i = 0; i < output.lengthOf(); i++) {
        auto value = output.e<Nd4jLong>(i);
        ASSERT_TRUE(value >= 0 && value < ClassValue);
        double* z = counts.bufferAsT<double>();
        z[value] += 1;
    }

    for (int i = 0; i < counts.lengthOf(); i++) {
        auto c = counts.e<double>(i);
        auto p = probExpect.e<double>(i);
        // printf("Get freq : %f  Expect freq: %f \n", c / Samples, p);
        ASSERT_NEAR((c / Samples), p, 3e-3);
    }

    deviation = output.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    mean = output.meanNumber();
    // printf("Var: %f  Mean: %f \n", deviation.e<double>(0), mean.e<double>(0));
    ASSERT_NEAR(1.2175, deviation.e<double>(0), 3e-3);
    ASSERT_NEAR(2.906, mean.e<double>(0), 3e-3);
}
