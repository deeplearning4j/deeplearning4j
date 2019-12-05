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
// Created by GS <sgazeos@gmail.com> on 22.07.2019.
//

#include "testlayers.h"
#include <NDArray.h>
#include <ShapeUtils.h>
#include <reduce3.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>
#include <type_conversions.h>
#include <ops/declarable/CustomOperations.h>
using namespace nd4j;
using namespace nd4j::ops;

class NativeOpsTests : public testing::Test {
public:

};


TEST_F(NativeOpsTests, CreateContextTests_1) {
//    auto x = NDArrayFactory::create<float>('c', {5, 5});
//    x.assign(1.0);
//    auto z = NDArrayFactory::create<float>('c', {5,5});
//    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    auto context = ::createContext();
    ASSERT_TRUE(context == nullptr);
    //delete context;
}

TEST_F(NativeOpsTests, CreateContextTests_2) {
//    auto x = NDArrayFactory::create<float>('c', {5, 5});
//    x.assign(1.0);
//    auto z = NDArrayFactory::create<float>('c', {5,5});
//    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    auto context1 = ::createContext();
    auto context2 = ::createContext();
    ASSERT_TRUE(context1 == context2);
    //delete context1;
    //delete context2;
}

TEST_F(NativeOpsTests, PointerTests_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1,2,3,4,5});
//    x.linspace(1.0);
#ifdef __CUDABLAS__
printf("Unsupported for cuda now.\n");
#else
    ::tryPointer(nullptr, x.getBuffer(), 4);
#endif

//    auto exp = NDArrayFactory::create<float>('c', {5, 5});
//    exp.assign(-1.0);
//
//    nd4j::ops::LegacyTransformSameOp op(transform::Neg); // Neg
//    auto result = op.execute({&x}, {}, {});
//
//    ASSERT_EQ(1, result->size());
//
//    auto z = result->at(0);
//
//    ASSERT_TRUE(exp.equalsTo(z));
//
//    delete result;
}

TEST_F(NativeOpsTests, ThresholdTests_1) {
//    auto x = NDArrayFactory::create<float>('c', {5}, {1,2,3,4,5});
//    x.linspace(1.0);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    ::setElementThreshold(4);
    ASSERT_TRUE(4 == nd4j::Environment::getInstance()->elementwiseThreshold());
#endif

}

TEST_F(NativeOpsTests, ThresholdTests_2) {
//    auto x = NDArrayFactory::create<float>('c', {5}, {1,2,3,4,5});
//    x.linspace(1.0);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    ::setTADThreshold(4);
    ASSERT_TRUE(4 == nd4j::Environment::getInstance()->tadThreshold());
#endif

}

TEST_F(NativeOpsTests, ExecIndexReduce_1) {
    auto x = NDArrayFactory::create<float>('c', {5}, {1,2,3,4,5});
    auto exp = NDArrayFactory::create<Nd4jLong>(120);
    x.linspace(1.0);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    ::execIndexReduceScalar(nullptr,
                                        indexreduce::IndexMax,
                                        x.buffer(), x.shapeInfo(),
                                        nullptr, nullptr,
                                        nullptr,
                                        exp.buffer(), exp.shapeInfo(),
                                        nullptr, nullptr);

    ASSERT_TRUE(exp.e<Nd4jLong>(0) == 4LL);
#endif

}

TEST_F(NativeOpsTests, ExecIndexReduce_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>(120);
    x.linspace(1.0);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    NDArray dimension = NDArrayFactory::create<int>({});
    ::execIndexReduce(nullptr,
                                   indexreduce::IndexMax,
                                   x.buffer(), x.shapeInfo(),
                                   nullptr, nullptr,
                                   nullptr,
                                   exp.buffer(), exp.shapeInfo(),
                                   nullptr, nullptr,
                                   dimension.buffer(), dimension.shapeInfo(),
                                   nullptr, nullptr);

    ASSERT_TRUE(exp.e<Nd4jLong>(0) == 24LL);
#endif

}

TEST_F(NativeOpsTests, ExecBroadcast_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 1});
    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    x.linspace(1.0);
    y.linspace(2,2);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    auto dimension = NDArrayFactory::create<int>('c', {1}, {1});

    ::execBroadcast(nullptr,
                             broadcast::Add,
                             x.buffer(), x.shapeInfo(),
                             nullptr, nullptr,
                             y.buffer(), y.shapeInfo(),
                             nullptr, nullptr,
                             exp.buffer(), exp.shapeInfo(),
                             nullptr, nullptr,
                             dimension.buffer(), dimension.shapeInfo(),
                             nullptr, nullptr);

    ASSERT_TRUE(exp.e<float>(0) == 3.);
#endif

}

TEST_F(NativeOpsTests, ExecBroadcast_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 1});
    auto exp = NDArrayFactory::create<bool>('c', {5, 5});
    x.linspace(1.0);
    y.linspace(2,2);
#ifdef __CUDABLAS__
printf("Unsupported for cuda now.\n");
#else
    int dimd = 0;
    auto dimension = NDArrayFactory::create<int>('c', {1}, {dimd});

    ::execBroadcastBool(nullptr,
        broadcast::EqualTo,
        x.buffer(), x.shapeInfo(),
        nullptr, nullptr,
        y.buffer(), y.shapeInfo(),
        nullptr, nullptr,
        exp.buffer(), exp.shapeInfo(),
        nullptr, nullptr,
        nullptr,
        dimension.buffer(), dimension.shapeInfo(),
        nullptr, nullptr);
        ASSERT_TRUE(exp.e<bool>(1) && !exp.e<bool>(0));
#endif

}

TEST_F(NativeOpsTests, ExecPairwise_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    x.linspace(1.0);
    y.assign(2.);
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    ::execPairwiseTransform(nullptr,
                               pairwise::Add,
                               x.buffer(), x.shapeInfo(),
                               nullptr, nullptr,
                               y.buffer(), y.shapeInfo(),
                               nullptr, nullptr,
                               exp.buffer(), exp.shapeInfo(),
                               nullptr, nullptr,
                               nullptr);
    ASSERT_TRUE(exp.e<float>(5) == 8.);
#endif

}

TEST_F(NativeOpsTests, ExecPairwise_2) {
    auto x = NDArrayFactory::create<bool>('c', {5, 5});
    auto y = NDArrayFactory::create<bool>('c', {5, 5});
    auto exp = NDArrayFactory::create<bool>('c', {5, 5});
    x.assign(true);
    y.assign(false);
    y.t<bool>(5) = true;
#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    ::execPairwiseTransformBool(nullptr,
                                   pairwise::And,
                                   x.buffer(), x.shapeInfo(),
                                   nullptr, nullptr,
                                   y.buffer(), y.shapeInfo(),
                                   nullptr, nullptr,
                                   exp.buffer(), exp.shapeInfo(),
                                   nullptr, nullptr,
                                   nullptr);
    ASSERT_TRUE(exp.e<bool>(5) && !exp.e<bool>(4));
#endif

}

TEST_F(NativeOpsTests, ReduceTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    auto dimension = NDArrayFactory::create<int>('c', {1}, {1});

    ::execReduceFloat(nullptr,
                           reduce::Mean,
                           x.buffer(), x.shapeInfo(),
                           nullptr, nullptr,
                           nullptr,
                           exp.buffer(), exp.shapeInfo(),
                           nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce Mean");
    ASSERT_TRUE(exp.e<float>(0) == 13.);
#endif

}

TEST_F(NativeOpsTests, ReduceTest_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else

    ::execReduceSame(nullptr,
                             reduce::Sum,
                             x.buffer(), x.shapeInfo(),
                             nullptr, nullptr,
                             nullptr,
                             exp.buffer(), exp.shapeInfo(),
                             nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce Sum");
    ASSERT_TRUE(exp.e<float>(0) == 325.);
#endif

}

TEST_F(NativeOpsTests, ReduceTest_3) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<bool>(false);
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else

    ::execReduceBool(nullptr,
                            reduce::All,
                            x.buffer(), x.shapeInfo(),
                            nullptr, nullptr,
                            nullptr,
                            exp.buffer(), exp.shapeInfo(),
                            nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.e<bool>(0) == true);
#endif

}

TEST_F(NativeOpsTests, ReduceTest_4) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>(120LL);
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else

    ::execReduceLong(nullptr,
                            reduce::CountNonZero,
                            x.buffer(), x.shapeInfo(),
                            nullptr, nullptr,
                            nullptr,
                            exp.buffer(), exp.shapeInfo(),
                            nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce CountNonZero");
    ASSERT_TRUE(exp.e<Nd4jLong>(0) == 25LL);
#endif

}

TEST_F(NativeOpsTests, ReduceTest_5) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<Nd4jLong>(120LL);
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    auto dimension = NDArrayFactory::create<int>({0, 1});
    ::execReduceLong2(nullptr,
                            reduce::CountNonZero,
                            x.buffer(), x.shapeInfo(),
                            nullptr, nullptr,
                            nullptr,
                            exp.buffer(), exp.shapeInfo(),
                            nullptr, nullptr,
                            dimension.buffer(), dimension.shapeInfo(),
                            dimension.specialBuffer(), dimension.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce CountNonZero");
    ASSERT_TRUE(exp.e<Nd4jLong>(0) == 25LL);
#endif

}

TEST_F(NativeOpsTests, ReduceTest_6) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto z = NDArrayFactory::create<Nd4jLong>({5, 4, 3, 2, 1});
    auto exp = NDArrayFactory::create<Nd4jLong>({1,2,3,4,6});
    x.linspace(1.0);

#ifdef __CUDABLAS__
    printf("Unsupported for cuda now.\n");
#else
    auto dimension = NDArrayFactory::create<int>('c', {1}, {1});
    x.p(5, 0);
    x.p(10, 0); x.p(11, 0);
    x.p(15, 0); x.p(16, 0); x.p(17, 0);
    x.p(20, 0); x.p(21, 0); x.p(22, 0); x.p(23, 0);
    ::execReduceLong2(nullptr,
                            reduce::CountNonZero,
                            x.buffer(), x.shapeInfo(),
                            nullptr, nullptr,
                            nullptr,
                            exp.buffer(), exp.shapeInfo(),
                            nullptr, nullptr,
                            dimension.buffer(), dimension.shapeInfo(),
                            dimension.specialBuffer(), dimension.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce CountNonZero");
    ASSERT_TRUE(exp.equalsTo(z));
#endif

}

TEST_F(NativeOpsTests, ReduceTest_7) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    auto z = NDArrayFactory::create<float>(13.);


    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    x.syncToHost();
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
#endif
    x.linspace(1.0);
    x.syncToDevice();
    dimension.syncToHost();
    ::execReduceFloat2(extra,
                             reduce::Mean,
                             x.buffer(), x.shapeInfo(),
                             x.specialBuffer(), x.specialShapeInfo(),
                             nullptr,
                             exp.buffer(), exp.shapeInfo(),
                             exp.specialBuffer(), exp.specialShapeInfo(),
                             dimension.buffer(), dimension.shapeInfo(),
                             dimension.specialBuffer(), dimension.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce Mean");
    ASSERT_TRUE(exp.equalsTo(z));

}

TEST_F(NativeOpsTests, ReduceTest_8) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto z = NDArrayFactory::create<float>(120.);
    auto exp = NDArrayFactory::create<float>(325.);


    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
#endif
    x.linspace(1.0);
    x.syncToDevice();

    dimension.syncToHost();

    ::execReduceSame2(extra,
                            reduce::Sum,
                            x.buffer(), x.shapeInfo(),
                            x.specialBuffer(), x.specialShapeInfo(),
                            nullptr,
                            z.buffer(), z.shapeInfo(),
                            z.specialBuffer(), z.specialShapeInfo(),
                            dimension.buffer(), dimension.shapeInfo(),
                            dimension.specialBuffer(), dimension.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce Sum");
    ASSERT_TRUE(exp.equalsTo(z));

}

TEST_F(NativeOpsTests, ReduceTest_9) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<bool>(false);
    auto z = NDArrayFactory::create<bool>(true);

    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
#endif
    x.linspace(1.0);
    x.syncToDevice();

    dimension.syncToHost();
    ::execReduceBool2(extra,
                            reduce::All,
                            x.buffer(), x.shapeInfo(),
                            x.specialBuffer(), x.specialShapeInfo(),
                            nullptr,
                            exp.buffer(), exp.shapeInfo(),
                            exp.specialBuffer(), exp.specialShapeInfo(),
                            dimension.buffer(), dimension.shapeInfo(),
                            dimension.specialBuffer(), dimension.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, Reduce3Test_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    auto z = NDArrayFactory::create<float>(650.);

    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    y.assign(2.);
    x.syncToDevice();

    ::execReduce3(extra,
                            reduce3::Dot,
                            x.buffer(), x.shapeInfo(),
                            x.specialBuffer(), x.specialShapeInfo(),
                            nullptr,
                            y.buffer(), y.shapeInfo(),
                            y.specialBuffer(), y.specialShapeInfo(),
                            exp.buffer(), exp.shapeInfo(),
                            exp.specialBuffer(), exp.specialShapeInfo());
    //z.printIndexedBuffer("Z");
    //exp.printIndexedBuffer("Reduce3 Dot");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, Reduce3Test_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    auto z = NDArrayFactory::create<float>(650.);

    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    y.assign(2.);
    x.syncToDevice();

    ::execReduce3Scalar(extra,
                         reduce3::Dot,
                         x.buffer(), x.shapeInfo(),
                         x.specialBuffer(), x.specialShapeInfo(),
                         nullptr,
                         y.buffer(), y.shapeInfo(),
                         y.specialBuffer(), y.specialShapeInfo(),
                         exp.buffer(), exp.shapeInfo(),
                         exp.specialBuffer(), exp.specialShapeInfo());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce3 Dot");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, Reduce3Test_3) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    auto z = NDArrayFactory::create<float>(650.);

    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    y.assign(2.);
    x.syncToDevice();
    dimension.syncToHost();

    ::execReduce3Tad(extra,
                         reduce3::Dot,
                         x.buffer(), x.shapeInfo(),
                         x.specialBuffer(), x.specialShapeInfo(),
                         nullptr,
                         y.buffer(), y.shapeInfo(),
                         y.specialBuffer(), y.specialShapeInfo(),
                         exp.buffer(), exp.shapeInfo(),
                         exp.specialBuffer(), exp.specialShapeInfo(),
                         dimension.buffer(), dimension.shapeInfo(),
                         dimension.specialBuffer(), dimension.specialShapeInfo(),
                         nullptr, nullptr, nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, Reduce3Test_4) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>(120.);
    auto z = NDArrayFactory::create<float>(650.);

    auto dimension = NDArrayFactory::create<int>('c', {2}, {0, 1});
    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    y.assign(2.);
    x.syncToDevice();
    dimension.syncToHost();
    int* dimensions = reinterpret_cast<int*>(dimension.buffer());
    auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), dimensions, dimension.lengthOf());
    auto tadPackY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(y.shapeInfo(), dimensions, dimension.lengthOf());

    auto hTADShapeInfoX = tadPackX.primaryShapeInfo();
    auto hTADOffsetsX = tadPackX.primaryOffsets();
    auto hTADShapeInfoY = tadPackY.primaryShapeInfo();
    auto hTADOffsetsY = tadPackY.primaryOffsets();

    ::execReduce3All(extra,
                         reduce3::Dot,
                         x.buffer(), x.shapeInfo(),
                         x.specialBuffer(), x.specialShapeInfo(),
                         nullptr,
                         y.buffer(), y.shapeInfo(),
                         y.specialBuffer(), y.specialShapeInfo(),
                         exp.buffer(), exp.shapeInfo(),
                         exp.specialBuffer(), exp.specialShapeInfo(),
                         dimension.buffer(), dimension.shapeInfo(),
                         dimension.specialBuffer(), dimension.specialShapeInfo(),
                         hTADShapeInfoX, hTADOffsetsX, hTADShapeInfoY, hTADOffsetsY);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, ScalarTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>(10.);
    auto exp = NDArrayFactory::create<float>('c', {5,5});
    auto z = NDArrayFactory::create<float>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    y.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    z.linspace(10., 10.);
    //y.assign(2.);
    x.syncToDevice();
    z.syncToDevice();
    ::execScalar(extra,
                            scalar::Multiply,
                            x.buffer(), x.shapeInfo(),
                            x.specialBuffer(), x.specialShapeInfo(),
                            exp.buffer(), exp.shapeInfo(),
                            exp.specialBuffer(), exp.specialShapeInfo(),
                            y.buffer(), y.shapeInfo(),
                            y.specialBuffer(), y.specialShapeInfo(), nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, ScalarTest_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>(10.f);
    auto exp = NDArrayFactory::create<bool>('c', {5,5});
    auto z = NDArrayFactory::create<bool>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    y.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    z.assign(false);
    //y.assign(2.);
    x.syncToDevice();
    z.syncToDevice();
    ::execScalarBool(extra,
                        scalar::GreaterThan,
                        x.buffer(), x.shapeInfo(),
                        x.specialBuffer(), x.specialShapeInfo(),
                        exp.buffer(), exp.shapeInfo(),
                        exp.specialBuffer(), exp.specialShapeInfo(),
                        y.buffer(), y.shapeInfo(),
                        y.specialBuffer(), y.specialShapeInfo(), nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.e<bool>(5) == z.e<bool>(5) && exp.e<bool>(15) != z.e<bool>(15));
}

TEST_F(NativeOpsTests, SummaryStatsScalarTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5}, {0.1f, 0.2f, 0.3f, -0.3f, -0.5f, 0.5f, 0.7f, 0.9f, 0.8f, 0.1f, 0.11f, 0.12f, 0.5f, -0.8f, -0.9f, 0.4f, 0.1f, 0.2f, 0.3f, -0.3f, -0.5f, 0.2f, 0.3f, -0.3f, -0.5f});
    auto exp = NDArrayFactory::create<float>(0.9f);
    auto z = NDArrayFactory::create<float>(0.21587136f);

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    ::execSummaryStatsScalar(extra,
                        variance::SummaryStatsVariance,
                        x.buffer(), x.shapeInfo(),
                        x.specialBuffer(), x.specialShapeInfo(),
                        nullptr,
                        exp.buffer(), exp.shapeInfo(),
                        exp.specialBuffer(), exp.specialShapeInfo(), false);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Standard Variance");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, SummaryStatsScalarTest_2) {
    auto x = NDArrayFactory::create<double>('c', {5, 5}, {0.1, 0.2, 0.3, -0.3, -0.5, 0.5, 0.7, 0.9, 0.8, 0.1, 0.11, 0.12, 0.5, -0.8, -0.9, 0.4, 0.1, 0.2, 0.3, -0.3, -0.5, 0.2, 0.3, -0.3, -0.5});
    auto exp = NDArrayFactory::create<double>(0.9);
    auto z = NDArrayFactory::create<double>(0.21587136);

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    ::execSummaryStats(extra,
                                    variance::SummaryStatsVariance,
                                    x.buffer(), x.shapeInfo(),
                                    x.specialBuffer(), x.specialShapeInfo(),
                                    nullptr,
                                    exp.buffer(), exp.shapeInfo(),
                                    exp.specialBuffer(), exp.specialShapeInfo(), false);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Standard Variance");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, SummaryStatsScalarTest_3) {
    auto x = NDArrayFactory::create<double>('c', {5, 5}, {0.1, 0.2, 0.3, -0.3, -0.5, 0.5, 0.7, 0.9, 0.8, 0.1, 0.11, 0.12, 0.5, -0.8, -0.9, 0.4, 0.1, 0.2, 0.3, -0.3, -0.5, 0.2, 0.3, -0.3, -0.5});
    auto exp = NDArrayFactory::create<double>(0.9);
    auto z = NDArrayFactory::create<double>(0.21587136);

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    auto dimensions = NDArrayFactory::create<int>({0, 1});
    ::execSummaryStatsTad(extra,
                                    variance::SummaryStatsVariance,
                                    x.buffer(), x.shapeInfo(),
                                    x.specialBuffer(), x.specialShapeInfo(),
                                    nullptr,
                                    exp.buffer(), exp.shapeInfo(),
                                    exp.specialBuffer(), exp.specialShapeInfo(),
                                    dimensions.buffer(), dimensions.shapeInfo(),
                                    dimensions.specialBuffer(), dimensions.specialShapeInfo(),
                                    false,
                                    nullptr, nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Standard Variance");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, TransformTest_1) {
    auto x = NDArrayFactory::create<double>('c', {5, 5}, {1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625});
    auto exp = NDArrayFactory::create<double>('c', {5, 5});
    auto z = NDArrayFactory::create<double>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    z.linspace(1.);
    ::execTransformFloat(extra,
                              transform::Sqrt,
                              x.buffer(), x.shapeInfo(),
                              x.specialBuffer(), x.specialShapeInfo(),

                              exp.buffer(), exp.shapeInfo(),
                              exp.specialBuffer(), exp.specialShapeInfo(),
                              nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Sqrt is");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, TransformTest_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5}, {1.f, 4.f, 9.f, 16.f, 25.f, 36.f, 49.f, 64.f, 81.f, 100.f, 121.f, 144.f, 169.f, 196.f, 225.f, 256.f, 289.f, 324.f, 361.f, 400.f, 441.f, 484.f, 529.f, 576.f, 625.f});
    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    auto z = NDArrayFactory::create<float>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    z.linspace(1.);
    ::execTransformSame(extra,
                                transform::Square,
                                z.buffer(), z.shapeInfo(),
                                z.specialBuffer(), z.specialShapeInfo(),

                                exp.buffer(), exp.shapeInfo(),
                                exp.specialBuffer(), exp.specialShapeInfo(),
                                nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Square is");
    ASSERT_TRUE(exp.equalsTo(x));
}

TEST_F(NativeOpsTests, TransformTest_3) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<bool>('c', {5, 5});
    auto z = NDArrayFactory::create<bool>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.);
    z.assign(true);
    x.p(24, -25);
    z.p(24, false);
    ::execTransformBool(extra,
                                transform::IsPositive,
                                x.buffer(), x.shapeInfo(),
                                x.specialBuffer(), x.specialShapeInfo(),

                                exp.buffer(), exp.shapeInfo(),
                                exp.specialBuffer(), exp.specialShapeInfo(),
                                nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("IsPositive");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, TransformTest_4) {
    auto x = NDArrayFactory::create<double>('c', {5, 5}, {0, 1, 2, 3, 2, 1, 0, 1.57, 1.57, 1.57, 3.141592, 3.141592,
                                                         3.141592, 0, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0});
    auto exp = NDArrayFactory::create<double>('c', {5, 5});
    auto z = NDArrayFactory::create<double>('c', {5,5}, {1., 0.540302, -0.416147, -0.989992, -0.416147,  0.540302, 1.0,
                                                        0.000796, 0.000796, 0.000796, -1, -1, -1, 1., 1., 1.0, 1.0,
                                                        0.540302, 0.540302, -0.416147, -0.416147, -0.416147, 0.540302, 1., 1.});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    //z.linspace(1.);
    ::execTransformStrict(extra,
                                transform::Cosine,
                                x.buffer(), x.shapeInfo(),
                                x.specialBuffer(), x.specialShapeInfo(),

                                exp.buffer(), exp.shapeInfo(),
                                exp.specialBuffer(), exp.specialShapeInfo(),
                                nullptr);
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Cosine");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, ScalarTadTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>(10.f);
    auto exp = NDArrayFactory::create<float>('c', {5,5});
    auto z = NDArrayFactory::create<float>('c', {5,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    y.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    z.linspace(10., 10.);
    //y.assign(2.);
    x.syncToDevice();
    z.syncToDevice();
    auto dimension = NDArrayFactory::create<int>({0, 1});
    auto dimensions = reinterpret_cast<int*>(dimension.buffer());
    auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), dimensions, dimension.lengthOf());
    auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.shapeInfo(), dimensions, dimension.lengthOf());

    ::execScalarTad(extra,
                        scalar::Multiply,
                        x.buffer(), x.shapeInfo(),
                        x.specialBuffer(), x.specialShapeInfo(),
                        exp.buffer(), exp.shapeInfo(),
                        exp.specialBuffer(), exp.specialShapeInfo(),
                        y.buffer(), y.shapeInfo(),
                        y.specialBuffer(), y.specialShapeInfo(),
                        nullptr,
                        dimension.buffer(), dimension.shapeInfo(),
                        dimension.specialBuffer(), dimension.specialShapeInfo(),
                        tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("Reduce All");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, ScalarTadTest_2) {
    auto x = NDArrayFactory::create<bool>('c', {5, 5});
    auto y = NDArrayFactory::create<bool>(true);
    auto exp = NDArrayFactory::create<bool>('c', {5,5});
    auto z = NDArrayFactory::create<bool>('c', {5, 5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    y.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.assign(false);
    x.p(5, true);
    x.p(15, true);
    //z.linspace(10., 10.);
    //y.assign(2.);
    x.syncToDevice();
    z.syncToDevice();
    auto dimension = NDArrayFactory::create<int>({0, 1});
    auto dimensions = reinterpret_cast<int*>(dimension.buffer());
    auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), dimensions, dimension.lengthOf());
    auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.shapeInfo(), dimensions, dimension.lengthOf());
    z.assign(true);

    ::execScalarBoolTad(extra,
                        scalar::And,
                        x.buffer(), x.shapeInfo(),
                        x.specialBuffer(), x.specialShapeInfo(),
                        exp.buffer(), exp.shapeInfo(),
                        exp.specialBuffer(), exp.specialShapeInfo(),
                        y.buffer(), y.shapeInfo(),
                        y.specialBuffer(), y.specialShapeInfo(),
                        nullptr,
                        dimension.buffer(), dimension.shapeInfo(),
                        dimension.specialBuffer(), dimension.specialShapeInfo(),
                        tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(), tadPackZ.primaryShapeInfo(), tadPackZ.primaryOffsets());
//    x.printIndexedBuffer("Input");
//    exp.printIndexedBuffer("And");
    ASSERT_TRUE(exp.e<bool>(5) == z.e<bool>(5) && exp.e<bool>(15));
}

TEST_F(NativeOpsTests, ConcatTest_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {10,5});
    auto z = NDArrayFactory::create<float>('c', {10,5});

    Nd4jPointer extra[6];
#ifdef __CUDABLAS__
    extra[1] = x.getContext()->getCudaStream();
    extra[0] = extra[2] = extra[3] = extra[4] = extra[5] = nullptr;
    x.syncToHost();
    y.syncToHost();
    printf("Unsupported for CUDA platform yet.\n");
    return;
#endif
    x.linspace(1.0);
    y.linspace(26);

    //y.assign(2.);
    x.syncToDevice();
    z.syncToDevice();
    int d = 0;
    auto dimension = NDArrayFactory::create<int>('c', {1}, {d});
    auto dimensions = reinterpret_cast<int*>(dimension.buffer());
    //auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), dimensions, dimension.lengthOf());
    auto tadPackZ = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.shapeInfo(), dimensions, dimension.lengthOf());
    exp.linspace(1);
    Nd4jPointer datas[] = {x.buffer(), y.buffer()};
    Nd4jPointer shapes[] = {x.shapeInfo(), y.shapeInfo()};

    ::specialConcat(extra,
                    0, 2, datas, shapes, z.buffer(), z.shapeInfo(), nullptr, nullptr);

//    exp.printIndexedBuffer("Exp");
//    z.printIndexedBuffer("Concat");
    ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(NativeOpsTests, InitializeTest_1) {
//    ::initializeDevicesAndFunctions();
}

TEST_F(NativeOpsTests, MallocTest_1) {
    auto a = ::mallocHost(16, 0);
    ::freeHost(a);
    auto dA = ::mallocDevice(16, 0, 0);
    ::freeDevice(dA, 0);
}

TEST_F(NativeOpsTests, OMPTest_1) {
    auto maxThreads = ::ompGetMaxThreads();
    auto numThreads = ::ompGetNumThreads();
    //::setOmpMinThreads(maxThreads);
    //::setOmpNumThreads(numThreads);
}

TEST_F(NativeOpsTests, CreateTest_1) {
    auto xx = ::createContext();
    auto yy = ::createStream();
    auto zz = ::createEvent();
    ::destroyEvent(zz);
    if (xx)
        delete (LaunchContext*)xx;
    if (yy)
        printf("Stream should be destoyed before.");

}

TEST_F(NativeOpsTests, MemTest_1) {
    auto x = NDArrayFactory::create<double>({10, 20, 30, 40, 50});
    auto y = NDArrayFactory::create<double>({20, 20, 20, 20, 20});

#ifdef __CUDABLAS__
    return ;
#endif
    //ASSERT_TRUE(0 == ::memcpy(x.buffer(), y.buffer(), x.lengthOf() * sizeof(double), 0, nullptr));
    ASSERT_TRUE(0 == ::memcpyAsync(x.buffer(), y.buffer(), x.lengthOf() * sizeof(double), 0, nullptr));
    //ASSERT_TRUE(0 == ::memset(x.buffer(), 119, x.lengthOf() * sizeof(double), 0, nullptr));
    ASSERT_TRUE(0 == ::memsetAsync(x.buffer(), 119, x.lengthOf() * sizeof(double), 0, nullptr));

}

TEST_F(NativeOpsTests, PullRowsTest_1) {
    NDArray x('c', {5, 1}, {0,1,2,3,4});
    NDArray z('c', {4, 1}, nd4j::DataType::DOUBLE);
    NDArray exp('c', {4, 1}, {0,2,3,4});

    Nd4jLong indexes[] = {0,2,3,4};
    PointersManager pm(LaunchContext::defaultContext(), "NativeOpsTests::pullRows");
    auto pidx = reinterpret_cast<Nd4jLong *>(pm.replicatePointer(indexes, 4 * sizeof(Nd4jLong)));

    std::vector<int> dims = {1};

    auto xTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.getShapeInfo(), dims);
    auto zTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(z.getShapeInfo(), dims);

    Nd4jPointer nativeStart[2];

#ifdef __CUDABLAS__
    nativeStart[1] = (x.getContext()->getCudaStream());
#endif

    pullRows(nativeStart, x.buffer(), x.getShapeInfo(), x.getSpecialBuffer(), x.getSpecialShapeInfo(),
                z.buffer(), z.getShapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                4, pidx,
                xTadPack.platformShapeInfo(), xTadPack.platformOffsets(),
                zTadPack.platformShapeInfo(), zTadPack.platformOffsets());

    ASSERT_TRUE(z.equalsTo(exp));
    pm.synchronize();
}

TEST_F(NativeOpsTests, TadPackTest_1) {
    int dimension[] = {1};
    int const dimensionLength = 1;
    auto x = NDArrayFactory::create<int>('c', {2,3,4});
    nd4j::TadPack* pack = ::tadOnlyShapeInfo(x.shapeInfo(),
                                    dimension,
                                    dimensionLength);
    ASSERT_TRUE(pack != nullptr);
    delete pack;
}

TEST_F(NativeOpsTests, AverageTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {5,5});
    auto z = NDArrayFactory::create<float>('c', {5,5});
#ifdef __CUDABLAS__
    return;
#endif
    x.linspace(1);
    exp.linspace(1);
    Nd4jPointer xList[] = {x.buffer(), x.buffer()};
    Nd4jPointer dxList[] = {x.specialBuffer(), x.specialBuffer()};
    ::average(nullptr,
                            xList, x.shapeInfo(),
                            dxList, x.specialShapeInfo(),
                            z.buffer(), z.shapeInfo(),
                            z.specialBuffer(), z.specialShapeInfo(),
                            2,
                            x.lengthOf(),
                            true);
//    z.printIndexedBuffer("RES");
    ASSERT_TRUE(z.equalsTo(exp));
}

TEST_F(NativeOpsTests, AccumulateTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {5,5});
    auto z = NDArrayFactory::create<float>('c', {5,5});
#ifdef __CUDABLAS__
    return;
#endif
    x.linspace(1);
    exp.linspace(2,2);
    Nd4jPointer xList[] = {x.buffer(), x.buffer()};
    Nd4jPointer dxList[] = {x.specialBuffer(), x.specialBuffer()};
    ::accumulate(nullptr,
                     xList, x.shapeInfo(),
                     dxList, x.specialShapeInfo(),
                     z.buffer(), z.shapeInfo(),
                     z.specialBuffer(), z.specialShapeInfo(),
                     2,
                     x.lengthOf());
//    z.printIndexedBuffer("RES");
    ASSERT_TRUE(z.equalsTo(exp));
}

TEST_F(NativeOpsTests, P2PTest_1) {
    ::enableP2P(true);
    ::checkP2P();
    ::isP2PAvailable();
}

TEST_F(NativeOpsTests, ShuffleTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {5,5});
    auto z = NDArrayFactory::create<float>('c', {5,5});
#ifdef __CUDABLAS__
    return;
#endif
    x.linspace(1);
    y.linspace(34);
    exp.linspace(2,2);
    Nd4jPointer xList[] = {x.buffer(), x.buffer()};
    Nd4jPointer dxList[] = {x.specialBuffer(), y.specialBuffer()};
    Nd4jPointer xShapeList[] = {x.shapeInfo(), y.shapeInfo()};
    Nd4jPointer dxShapeList[] = {x.specialShapeInfo(), y.specialShapeInfo()};
    Nd4jPointer zList[] = {z.buffer(), z.buffer()};
    Nd4jPointer dzList[] = {z.specialBuffer(), z.specialBuffer()};
    Nd4jPointer zShapeList[] = {z.shapeInfo(), z.shapeInfo()};
    Nd4jPointer dzShapeList[] = {z.specialShapeInfo(), z.specialShapeInfo()};
    int shuffleMap[] = {1, 0, 4, 3, 2};
    auto zTadPack = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.getShapeInfo(), {1});
    Nd4jPointer zListOffset[] = {zTadPack.platformOffsets(), zTadPack.platformOffsets()};
    Nd4jPointer zListTADs[] = {zTadPack.platformShapeInfo(), zTadPack.platformShapeInfo()};
    ::shuffle(nullptr,
                        xList, xShapeList,
                        dxList, dxShapeList,
                        zList, zShapeList,
                        dzList, dzShapeList,
                        2,
                        shuffleMap, zListTADs, zListOffset);
//    z.printIndexedBuffer("RES");
//    x.printIndexedBuffer("INPUT shuffled");
//    y.printIndexedBuffer("INPUT 2 shuffled");
//    ASSERT_TRUE(z.equalsTo(exp));
}

TEST_F(NativeOpsTests, ConvertTypesTest_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});

    auto exp = NDArrayFactory::create<double>('c', {5, 5});
    auto z = NDArrayFactory::create<double>('c', {5, 5});

#ifdef __CUDABLAS__
    return;
#endif
    x.linspace(2, 2);
    exp.linspace(2, 2);
    ::convertTypes(nullptr, ND4J_FLOAT32, x.buffer(), x.lengthOf(), ND4J_DOUBLE, z.buffer());
    ASSERT_TRUE(z.equalsTo(exp));
}

//TEST_F(NativeOpsTests, Test_Aggregations_1) {
//    NativeOps ops;
//    auto x = NDArrayFactory::create<float>('c', {5,5});
//    auto y = NDArrayFactory::create<float>('c', {5,5});
//
//
//    ops.execAggregate(nullptr, 0, maxArgs, maxShapes, maxIntArrays, maxIntArraySize, maxIndexArguments, maxRealArguments, pointer.data(), nd4j::DataType::FLOAT32);
//    void **arguments,
//    int numArguments,
//    Nd4jLong **shapeArguments,
//    int numShapeArguments,
//    int *indexArguments,
//    int numIndexArguments,
//    int **intArrays,
//    int numIntArrays,
//    void *realArguments,
//    int numRealArguments,
//    nd4j::DataType dtype
//}

TEST_F(NativeOpsTests, RandomTest_1) {
    auto z = NDArrayFactory::create<double>('c', {100});
    Nd4jPointer extra[] = {nullptr, nullptr};
#ifdef __CUDABLAS__
    return;
    extra[1] = z.getContext()->getCudaStream();
#endif
    graph::RandomGenerator rng(1023, 119);
    double p = 0.5;
    ::execRandom(extra, random::BernoulliDistribution, &rng, z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), &p);
}

TEST_F(NativeOpsTests, RandomTest_2) {
    auto x = NDArrayFactory::create<double>('c', {100});
    auto z = NDArrayFactory::create<double>('c', {100});
    Nd4jPointer extra[] = {nullptr, nullptr};
#ifdef __CUDABLAS__
    return;
    extra[1] = z.getContext()->getCudaStream();
#endif
    x.linspace(0, 0.01);
    graph::RandomGenerator rng(1023, 119);
    double p = 0.5;
    ::execRandom2(extra, random::DropOut, &rng, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), &p);
}

TEST_F(NativeOpsTests, RandomTest_3) {
    auto x = NDArrayFactory::create<double>('c', {100});
    auto y = NDArrayFactory::create<double>('c', {100});
    auto z = NDArrayFactory::create<double>('c', {100});
    Nd4jPointer extra[] = {nullptr, nullptr};
#ifdef __CUDABLAS__
    return;
    extra[1] = z.getContext()->getCudaStream();
#endif
    x.linspace(0, 0.01);
    x.linspace(1, -0.01);
    graph::RandomGenerator rng(1023, 119);
    double p = 0.5;
    ::execRandom3(extra, random::ProbablisticMerge, &rng, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), &p);
}

TEST_F(NativeOpsTests, RandomTest_4) {
#ifdef __CUDABLAS__
    return ;
#endif
    graph::RandomGenerator* rng = (graph::RandomGenerator*)::initRandom(nullptr, 1023, 0, nullptr);
    ::refreshBuffer(nullptr, 1203L, rng);
    ::reSeedBuffer(nullptr, 3113L, rng);
    ::destroyRandom(rng);
}

TEST_F(NativeOpsTests, SortTest_1) {
#ifdef __CUDABLAS__
    return ;
#endif
    auto sortedVals = NDArrayFactory::create<int>(
            {10, 1, 5, 120, 34, 5, 78, 138, 3, 111, 331, 29, 91, 71, 73, 50, 56, 4});
    auto exp = NDArrayFactory::create<int>({1, 3, 4, 5, 5, 10, 29, 34, 50, 56, 71, 73, 78, 91, 111, 120, 138, 331});

    ::sort(nullptr, sortedVals.buffer(), sortedVals.shapeInfo(), sortedVals.specialBuffer(),
             sortedVals.specialShapeInfo(), false);
    ASSERT_TRUE(sortedVals.equalsTo(exp));
}

TEST_F(NativeOpsTests, SortTests_2) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});
    Nd4jPointer extras[2];
#ifdef __CUDABLAS__
    extras[1] = LaunchContext::defaultContext()->getCudaStream();
#endif

    ::sortByKey(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(NativeOpsTests, SortTest_3) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

#ifdef __CUDABLAS__
    Nd4jPointer extras[2] = {nullptr, LaunchContext::defaultContext()->getCudaStream()};
#else
    Nd4jPointer extras[2];
#endif

    ::sortByKey(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(NativeOpsTests, SortTest_4) {
#ifdef __CUDABLAS__
    return ;
#endif
    auto sortedVals = NDArrayFactory::create<int>('c', {3, 6},
            { 10,   1, 5, 120,  34,  5,
                   78, 138,  3, 111, 331, 29,
                   91,  71, 73,  50,  56,  4});
    auto exp = NDArrayFactory::create<int>('c', {3, 6}, {1, 5, 5, 10, 34, 120, 3, 29, 78, 111, 138, 331, 4, 50, 56, 71, 73, 91});

    std::vector<int> dims({1});
    auto packX = ConstantTadHelper::getInstance()->tadForDimensions(sortedVals.shapeInfo(), {1});
    ::sortTad(nullptr, sortedVals.buffer(), sortedVals.shapeInfo(), sortedVals.specialBuffer(),
             sortedVals.specialShapeInfo(), dims.data(), dims.size(), packX.platformShapeInfo(), packX.platformOffsets(), false);
//    sortedVals.printBuffer("OUT");
//    exp.printIndexedBuffer("EXP");
    ASSERT_TRUE(sortedVals.equalsTo(exp));
}

TEST_F(NativeOpsTests, SortTests_5) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2];
#ifdef __CUDABLAS__
    extras[1] = LaunchContext::defaultContext()->getCudaStream();
#endif

    int axis = 1;

    ::sortTadByKey(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);
    k.tickWriteDevice();
    v.tickWriteDevice();

//    k.printIndexedBuffer("k");
//    v.printIndexedBuffer("v");

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

TEST_F(NativeOpsTests, SortTests_6) {
    auto k = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {1, 3, 5, 9, 0, 2, 4, 6, 7, 8,   1, 3, 5, 9, 0, 2, 4, 6, 7, 8});
    auto v = NDArrayFactory::create<double>('c', {2, 10}, {1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5,   1.5, 3.5, 5.5, 9.5, 0.5, 2.5, 4.5, 6.5, 7.5, 8.5});

    auto ek = NDArrayFactory::create<Nd4jLong>('c', {2, 10}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,     0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto ev = NDArrayFactory::create<double>('c', {2, 10}, {0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5,     0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5});

    Nd4jPointer extras[2];
#ifdef __CUDABLAS__
    extras[1] = LaunchContext::defaultContext()->getCudaStream();
#endif

    int axis = 1;

    ::sortTadByValue(extras, k.buffer(), k.shapeInfo(), k.specialBuffer(), k.specialShapeInfo(), v.buffer(), v.shapeInfo(), v.specialBuffer(), v.specialShapeInfo(), &axis, 1, false);
    k.tickWriteDevice();
    v.tickWriteDevice();

    ASSERT_EQ(ek, k);
    ASSERT_EQ(ev, v);
}

//TEST_F(NativeOpsTests, MapTests_1) {
//#ifdef __CUDABLAS__
//    return ;
//#endif
//#ifdef GTEST_OS_LINUX
//    auto ptrMap = ::mmapFile(nullptr, "/tmp/maptest.$$$", 100LL);
//
//    ::munmapFile(nullptr, ptrMap, 100LL);
//#endif
//
//}

TEST_F(NativeOpsTests, MapTests_1) {
    //printf("Custom ops: %s\n", ::getAllCustomOps());
    //printf("All ops: %s\n", ::getAllOperations());

    ::getAllCustomOps();
    ::getAllOperations();
}

TEST_F(NativeOpsTests, CustomOpTest_1) {
    auto x = NDArrayFactory::create<float>('c', {1, 6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto z = NDArrayFactory::create<float>('c', {6});
    auto e = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});

    nd4j::ops::squeeze op;

    Nd4jPointer ptrsInBuffer[] = {(Nd4jPointer) x.getBuffer(), x.getSpecialBuffer()};
    Nd4jPointer ptrsInShapes[] = {(Nd4jPointer) x.getShapeInfo(), x.getSpecialShapeInfo()};

    Nd4jPointer ptrsOutBuffers[] = {(Nd4jPointer) z.getBuffer(), z.getSpecialBuffer()};
    Nd4jPointer ptrsOutShapes[] = {(Nd4jPointer) z.getShapeInfo(), z.getSpecialShapeInfo()};


    auto status = ::execCustomOp(nullptr, op.getOpHash(), ptrsInBuffer, ptrsInShapes, 1, ptrsOutBuffers, ptrsOutShapes, 1, nullptr, 0, nullptr, 0, nullptr, 0, false);
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}
TEST_F(NativeOpsTests, CustomOpTests_2) {
    auto array0 = NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto array1 = NDArrayFactory::create<float>('c', {3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    auto z = NDArrayFactory::create<float>('c', {3, 2});

    auto exp = NDArrayFactory::create<float>('c', {3, 2}, {2.f, 4.f, 6.f, 8.f, 10.f, 12.f});
    Context ctx(1);

    NDArray::prepareSpecialUse({&z}, {&array0, &array1});

    ctx.setInputArray(0, array0.buffer(), array0.shapeInfo(), array0.getSpecialBuffer(), array0.getSpecialShapeInfo());
    ctx.setInputArray(1, array1.buffer(), array1.shapeInfo(), array1.getSpecialBuffer(), array1.getSpecialShapeInfo());
    ctx.setOutputArray(0, z.buffer(),           z.shapeInfo(), z.getSpecialBuffer(), z.getSpecialShapeInfo());

    ASSERT_EQ(2, ctx.width());

    nd4j::ops::add op;
    ::execCustomOp2(nullptr, op.getOpHash(), &ctx);

    NDArray::registerSpecialUse({&z}, {&array0, &array1});

    ASSERT_EQ(exp, z);
}
TEST_F(NativeOpsTests, CalculateOutputShapeTests_1) {
    auto input = NDArrayFactory::create<float>('c', {1, 2, 5, 4});
    auto weights = NDArrayFactory::create<float>('c', {2, 2, 2, 3});
    auto exp = NDArrayFactory::create<float>('c', {1, 3, 5, 4});

    nd4j::ops::conv2d op;

    std::vector<double> tArgs({});
    std::vector<Nd4jLong> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1});

    Nd4jPointer ptrs[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weights.getShapeInfo()};
#ifdef __CUDABLAS__
    return;
#endif

    auto shapeList = ::calculateOutputShapes(nullptr, op.getOpHash(), ptrs, 2, tArgs.data(), tArgs.size(), iArgs.data(), iArgs.size());

    ASSERT_EQ(1, shapeList->size());

    ASSERT_EQ(exp.rankOf(), shape::rank((Nd4jLong *)shapeList->at(0)));
    ASSERT_EQ(exp.sizeAt(0), shape::shapeOf((Nd4jLong *)shapeList->at(0))[0]);
    ASSERT_EQ(exp.sizeAt(1), shape::shapeOf((Nd4jLong *)shapeList->at(0))[1]);
    ASSERT_EQ(exp.sizeAt(2), shape::shapeOf((Nd4jLong *)shapeList->at(0))[2]);
    ASSERT_EQ(exp.sizeAt(3), shape::shapeOf((Nd4jLong *)shapeList->at(0))[3]);

    //int *ptr = (int *) shapeList[0];
    //delete[] ptr;
    //delete shapeList;

    ::deleteShapeList((Nd4jPointer) shapeList);
}

TEST_F(NativeOpsTests, CalculateOutputShapeTests_2) {
    auto input = NDArrayFactory::create<float>('c', {1, 2, 5, 4});
    auto weights = NDArrayFactory::create<float>('c', {2, 2, 2, 3});
    auto exp = NDArrayFactory::create<float>('c', {1, 3, 5, 4});

    nd4j::ops::conv2d op;

    std::vector<double> tArgs({});
    std::vector<bool> bArgsF({});
    std::vector<Nd4jLong> iArgs({2, 2, 1, 1, 0, 0, 1, 1, 1});

    Nd4jPointer shapePtrs[] = {(Nd4jPointer) input.getShapeInfo(), (Nd4jPointer) weights.getShapeInfo()};
    Nd4jPointer dataPtrs[] = {(Nd4jPointer)input.buffer(), (Nd4jPointer)weights.buffer()};
#ifdef __CUDABLAS__
    return;
#endif

    auto shapeList = ::calculateOutputShapes2(nullptr, op.getOpHash(), dataPtrs, shapePtrs, 2, const_cast<double*>(tArgs.data()), tArgs.size(),
                                                     const_cast<Nd4jLong*>(iArgs.data()), iArgs.size(), nullptr, bArgsF.size());
//                               Nd4jPointer* extraPointers, Nd4jLong hash, Nd4jPointer* inputBuffers, Nd4jPointer* inputShapes, int numInputShapes, double* tArgs, int numTArgs, Nd4jLong *iArgs, int numIArgs, bool *bArgs, int numBArgs
    ASSERT_EQ(1, shapeList->size());

    ASSERT_EQ(exp.rankOf(), shape::rank((Nd4jLong *)shapeList->at(0)));
    ASSERT_EQ(exp.sizeAt(0), shape::shapeOf((Nd4jLong *)shapeList->at(0))[0]);
    ASSERT_EQ(exp.sizeAt(1), shape::shapeOf((Nd4jLong *)shapeList->at(0))[1]);
    ASSERT_EQ(exp.sizeAt(2), shape::shapeOf((Nd4jLong *)shapeList->at(0))[2]);
    ASSERT_EQ(exp.sizeAt(3), shape::shapeOf((Nd4jLong *)shapeList->at(0))[3]);

    //int *ptr = (int *) shapeList[0];
    //delete[] ptr;
    //delete shapeList;

    ::deleteShapeList((Nd4jPointer) shapeList);
}

//Uncomment when needed only - massive calculations
//TEST_F(NativeOpsTests, BenchmarkTests_1) {
//
//    printf("%s\n", ::runLightBenchmarkSuit(true));
//    printf("%s\n", ::runFullBenchmarkSuit(true));
//}
