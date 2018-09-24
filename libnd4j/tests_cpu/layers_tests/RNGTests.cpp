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
    NativeOps nativeOps;
    Nd4jLong *_bufferA;
    Nd4jLong *_bufferB;

public:
    long _seed = 119L;
    nd4j::random::RandomBuffer *_rngA;
    nd4j::random::RandomBuffer *_rngB;

    NDArray* nexp0 = NDArrayFactory::create_<float>('c', {10, 10});
    NDArray* nexp1 = NDArrayFactory::create_<float>('c', {10, 10});
    NDArray* nexp2 = NDArrayFactory::create_<float>('c', {10, 10});

    RNGTests() {
        _bufferA = new Nd4jLong[100000];
        _bufferB = new Nd4jLong[100000];
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
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    x0.linspace(1);
    x1.linspace(1);

    float prob[] = {0.5f};

    //x0.applyRandom(random::DropOut, _rngA, nullptr, &x0, prob);
    //x1.applyRandom(random::DropOut, _rngB, nullptr, &x1, prob);

    ASSERT_TRUE(x0.equalsTo(&x1));

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

    ASSERT_TRUE(x0.equalsTo(&x1));

    // this check is required to ensure we're calling wrong signature
    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyDropOut(_rngA, &x0, 0.5f);
    RandomLauncher::applyDropOut(_rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_2) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyInvertedDropOut(_rngA, &x0, 0.5f);
    RandomLauncher::applyInvertedDropOut(_rngB, &x1, 0.5f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Launcher_3) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::applyAlphaDropOut(_rngA, &x0, 0.5f, 0.2f, 0.1f, 0.3f);
    RandomLauncher::applyAlphaDropOut(_rngB, &x1, 0.5f, 0.2f, 0.1f, 0.3f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Uniform_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillUniform(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillUniform(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));

    for (int e = 0; e < x0.lengthOf(); e++) {
        float v = x0.e<float>(e);
        ASSERT_TRUE(v >= 1.0f && v <= 2.0f);
    }
}

TEST_F(RNGTests, Test_Bernoulli_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBernoulli(_rngA, &x0, 1.0f);
    RandomLauncher::fillBernoulli(_rngB, &x1, 1.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Gaussian_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillGaussian(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillGaussian(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_LogNormal_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillLogNormal(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillLogNormal(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}

TEST_F(RNGTests, Test_Truncated_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillTruncatedNormal(_rngA, &x0, 1.0f, 2.0f);
    RandomLauncher::fillTruncatedNormal(_rngB, &x1, 1.0f, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Binomial_1) {
    auto x0 = NDArrayFactory::create<float>('c', {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBinomial(_rngA, &x0, 3, 2.0f);
    RandomLauncher::fillBinomial(_rngB, &x1, 3, 2.0f);

    ASSERT_TRUE(x0.equalsTo(&x1));

    //nexp2->printIndexedBuffer("nexp2");
    //x0.printIndexedBuffer("x0");

    ASSERT_FALSE(x0.equalsTo(nexp0));
    ASSERT_FALSE(x0.equalsTo(nexp1));
    ASSERT_FALSE(x0.equalsTo(nexp2));
}


TEST_F(RNGTests, Test_Uniform_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillUniform(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(0);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_Gaussian_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillGaussian(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(6);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_LogNorm_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillLogNormal(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(10);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_TruncatedNorm_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillTruncatedNormal(_rngB, &x1, 1.0f, 2.0f);

    auto op = new nd4j::ops::LegacyRandomOp(11);
    auto result = op->execute(_rngA, {&input}, {1.0f, 2.0f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Binomial_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBinomial(_rngB, &x1, 3, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp(9);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {3});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}


TEST_F(RNGTests, Test_Bernoulli_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2}, {10, 10});
    auto x1 = NDArrayFactory::create<float>('c', {10, 10});

    RandomLauncher::fillBernoulli(_rngB, &x1, 0.5f);

    auto op = new nd4j::ops::LegacyRandomOp(7);
    auto result = op->execute(_rngA, {&input}, {0.5f}, {});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    auto z = result->at(0);

    ASSERT_TRUE(x1.isSameShape(z));
    ASSERT_TRUE(x1.equalsTo(z));

    delete op;
    delete result;
}

TEST_F(RNGTests, Test_GaussianDistribution_1) {
    auto x = NDArrayFactory::create<float>('c', {2}, {10, 10});
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
    auto x = NDArrayFactory::create<float>('c', {2}, {10, 10});
    auto exp0 = NDArrayFactory::create<float>('c', {10, 10});


    nd4j::ops::random_bernoulli op;
    auto result = op.execute({&x}, {0.5f}, {});
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
    auto x = NDArrayFactory::create<float>('c', {2}, {10, 10});
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
    auto x = NDArrayFactory::create<float>('c', {2}, {10, 10});
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
        static void fillList(Nd4jLong seed, int numberOfArrays, std::vector<Nd4jLong> &shape, std::vector<NDArray*> &list, nd4j::random::RandomBuffer *rng) {
            NativeOps ops;
            ops.refreshBuffer(nullptr, seed, reinterpret_cast<Nd4jPointer>(rng));
            
            for (int i = 0; i < numberOfArrays; i++) {
                auto array = NDArrayFactory::create_<double>('c', shape);

                nd4j::ops::randomuniform op;
                op.execute(rng, {array}, {array}, {0.0, 1.0}, {}, true);

                list.emplace_back(array);
            }
        };
    }
}

TEST_F(RNGTests, Test_Reproducibility_9) { 
    NativeOps ops;
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) ops.initRandom(nullptr, seed, bufferSize, buffer);

    const int length = 4000000;
    int *arrayE = new int[length];
    int *arrayT = new int[length];

    for (int e = 0; e < length; e++)
        arrayE[e] = rng->relativeInt(e);

    rng->rewindH(static_cast<Nd4jLong>(length));

    ops.refreshBuffer(nullptr, seed, reinterpret_cast<Nd4jPointer>(rng));

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

    ops.destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_Reproducibility_8) { 
    NativeOps ops;
    Nd4jLong seed = 123;

    std::vector<int> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) ops.initRandom(nullptr, seed, bufferSize, buffer);

    const int length = 4000000;
    int *arrayE = new int[length];
    int *arrayT = new int[length];

    for (int e = 0; e < length; e++)
        arrayE[e] = static_cast<int>(rng->relativeT<float>(e));

    rng->rewindH(static_cast<Nd4jLong>(length));

    ops.refreshBuffer(nullptr, seed, reinterpret_cast<Nd4jPointer>(rng));

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

    ops.destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_RandomBuffer_Half_1) {
    NativeOps ops;
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) ops.initRandom(nullptr, seed, bufferSize, buffer);

    auto r0 = rng->relativeT<float16>(12L);
    auto r1 = rng->relativeT<float16>(13L);

    ASSERT_NE(r0, r1);

    ops.destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_Reproducibility_1) {
    NativeOps ops;
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 28, 28};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) ops.initRandom(nullptr, seed, bufferSize, buffer);


    std::vector<NDArray*> expList;
    nd4j::tests::fillList(seed, 10, shape, expList, rng);

    for (int e = 0; e < 2; e++) {
        std::vector<NDArray *> trialList;
        nd4j::tests::fillList(seed, 10, shape, trialList, rng);

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

    ops.destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}

TEST_F(RNGTests, Test_Reproducibility_2) {
    NativeOps ops;
    Nd4jLong seed = 123;

    std::vector<Nd4jLong> shape = {32, 3, 64, 64};
    const int bufferSize = 10000;
    int64_t buffer[bufferSize];

    auto rng = (nd4j::random::RandomBuffer *) ops.initRandom(nullptr, seed, bufferSize, buffer);

    std::vector<NDArray*> expList;
    nd4j::tests::fillList(seed, 10, shape, expList, rng);

    for (int e = 0; e < 2; e++) {
        std::vector<NDArray*> trialList;
        nd4j::tests::fillList(seed, 10, shape, trialList, rng);

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

    ops.destroyRandom(reinterpret_cast<Nd4jPointer>(rng));
}