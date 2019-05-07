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
// Created by raver119 on 16.10.2017.
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

using namespace nd4j;
using namespace nd4j::ops;

class LegacyOpsTests : public testing::Test {

};


TEST_F(LegacyOpsTests, TransformTests_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);
    auto z = NDArrayFactory::create<float>('c', {5,5});
    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformSameOp op(transform::Neg); // Neg
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(status, ND4J_STATUS_OK);
    //z.printIndexedBuffer("Output NEG");
    ASSERT_TRUE(z.equalsTo(&exp));
}

TEST_F(LegacyOpsTests, TransformTests_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);

    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(-1.0);

    nd4j::ops::LegacyTransformSameOp op(transform::Neg); // Neg
    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests,  Reciprocal_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(2.0f);

    auto ethalon = NDArrayFactory::create<float>('c', {5, 5});
    ethalon.assign(0.5f);

    nd4j::ops::LegacyTransformSameOp op(transform::Reciprocal); // Reciprocal
    Nd4jStatus status = op.execute({&x}, {&x}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);
    ASSERT_TRUE(ethalon.equalsTo(&x));
    
}

TEST_F(LegacyOpsTests,  PWT_Tests_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(2.0);

    auto y = NDArrayFactory::create<float>('c', {5, 5});
    y.assign(3.0);

    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp op(pairwise::Multiply); // Multiply
    Nd4jStatus status = op.execute({&x, &y}, {&x}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    ASSERT_TRUE(exp.equalsTo(&x));


}

TEST_F(LegacyOpsTests,  PWT_Tests_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(2.0);

    auto y = NDArrayFactory::create<float>('c', {5, 5});
    y.assign(3.0);

    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(6.0);

    nd4j::ops::LegacyPairwiseTransformOp op(pairwise::Multiply); // Multiply
    auto result = op.execute({&x, &y}, {}, {});

    auto z = result->at(0);

    //z->printBuffer("Z");
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests, Scalar_Test_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(2.0);

    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(7.0);

    nd4j::ops::LegacyScalarOp op(scalar::Add);
    op.execute({&x}, {&x}, {5.0}, {}, {}); //

    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, Scalar_Test_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(2.0);

    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    exp.assign(7.0);

    auto y = NDArrayFactory::create<float>(5.0f);

    nd4j::ops::LegacyScalarOp op(scalar::Add, y);
    auto result = op.execute({&x}, {}, {});

    auto z = result->at(0);
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);
    int opNum = reduce::Sum;
    nd4j::ops::LegacyReduceSameOp op(opNum);

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);
    z->printBuffer("ReduceTest1");
    ASSERT_TRUE(z->isScalar());
    ASSERT_NEAR(x.sumNumber().e<float>(0), z->e<float>(0), 1e-5f);

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);

    nd4j::ops::LegacyReduceSameOp op(reduce::Sum);
    auto axis = NDArrayFactory::create<Nd4jLong>('c', {1}, {1});
    auto result = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    auto exp = x.reduceAlongDimension(reduce::Sum, {1});

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete result;
    delete exp;
}


TEST_F(LegacyOpsTests, ReduceTests_3) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {1,1}, {1});


    nd4j::ops::LegacyReduceSameOp op(reduce::Sum);
    auto result = op.execute({&x, &indices}, {}, {});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Sum,{1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));
    
    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_4) {
    auto x = NDArrayFactory::create<float>('c', {2, 3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {1, 1}, {1});


    nd4j::ops::LegacyReduceSameOp op(reduce::Sum);
    auto result = op.execute({&x, &indices}, {}, {}, {true});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Sum, {1}, true);
    indices.printShapeInfo("Indices shape");
    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    z->printIndexedBuffer("Output reduce 4");
    exp.printIndexedBuffer("Expected reduce 4");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}

TEST_F(LegacyOpsTests, ReduceTests_5) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);
    int opNum = reduce::Mean;
    nd4j::ops::LegacyReduceFloatOp op(opNum);

    ResultSet* result = op.execute({&x}, {}, {}, {}, false, nd4j::DataType::FLOAT32);

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);
    z->printBuffer("ReduceTest1");
    ASSERT_TRUE(z->isScalar());
    ASSERT_NEAR(x.meanNumber().e<float>(0), z->e<float>(0), 1e-5f);

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_6) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(1.0);
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    nd4j::ops::LegacyReduceFloatOp op(reduce::Mean);

    auto result = op.execute({&x, &axis}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    auto exp = x.reduceAlongDimension(reduce::Mean, {1});

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete result;
    delete exp;
}


TEST_F(LegacyOpsTests, ReduceTests_7) {
    auto x = NDArrayFactory::create<float>('c', {3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {1,1}, {1});


    nd4j::ops::LegacyReduceFloatOp op(reduce::Mean);
    auto result = op.execute({&x, &indices}, {}, {});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Mean,{1});

    ASSERT_EQ(ND4J_STATUS_OK, result->status());

    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(LegacyOpsTests, ReduceTests_8) {
    auto x = NDArrayFactory::create<float>('c', {2, 3, 5});
    x.linspace(1);
    auto indices = NDArrayFactory::create<int>('c', {1}, {1});


    nd4j::ops::LegacyReduceFloatOp op(reduce::Mean);
    auto result = op.execute({&x, &indices}, {}, {}, {true});
    auto z = result->at(0);
    auto exp = x.reduceAlongDims(reduce::Mean, {1}, true);

    ASSERT_EQ(ND4J_STATUS_OK, result->status());
    z->printIndexedBuffer("Reduce8 output");
    z->printShapeInfo("Reduce8 shape");
    exp.printShapeInfo("Reduce8 expected shape");
    ASSERT_TRUE(exp.isSameShape(z));
    ASSERT_TRUE(exp.equalsTo(z));

    delete result;
}


TEST_F(LegacyOpsTests, IndexReduceTests_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    x.linspace(1);

    nd4j::ops::LegacyIndexReduceOp op(indexreduce::IndexMax);

    auto result = op.execute({&x}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);

    ASSERT_TRUE(z->isScalar());
    ASSERT_EQ(24, z->e<int>(0));

    delete result;
}


TEST_F(LegacyOpsTests, IndexReduceTests_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto indices = NDArrayFactory::create<int>('c', {1}, {1});
    x.linspace(1);
    auto exp = NDArrayFactory::create<Nd4jLong>({4,4,4,4,4});
    nd4j::ops::LegacyIndexReduceOp op(indexreduce::IndexMax);

    auto result = op.execute({&x, &indices}, {}, {});

    ASSERT_EQ(1, result->size());

    auto z = result->at(0);
    z->printIndexedBuffer("Hello indexreduce2");
    ASSERT_TRUE(exp.equalsTo(z));
    //ASSERT_EQ(4, z->e<int>(0));
    //ASSERT_EQ(4, z->e<int>(1));
    //ASSERT_EQ(4, z->e<int>(2));
    //ASSERT_EQ(4, z->e<int>(3));
    //ASSERT_EQ(4, z->e<int>(4));

    delete result;
}

TEST_F(LegacyOpsTests, Test_IsMax_1) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 2, 2, 2});
    auto z = NDArrayFactory::create<double>('c', {2, 2, 2, 2, 2, 2});
    x.linspace(1.0);
    z.assign(-589);

    double extra[] = {1.0, 0.0};

    NativeOpExcutioner::execTransformAny(transform::IsMax, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), extra, nullptr, nullptr);

    // z.printIndexedBuffer("z");
    for (int e = 0; e < z.lengthOf(); e++) {
        ASSERT_TRUE(z.e<double>(e) >= 0);
    }
}

TEST_F(LegacyOpsTests, Test_IsMax_2) {
    auto x = NDArrayFactory::create<double>('c', {2, 2, 2, 2, 2, 2});
    auto z = NDArrayFactory::create<bool>('c', {2, 2, 2, 2, 2, 2});
    x.linspace(1.0);
    z.assign(false);

    double extra[] = {1.0, 0.0};

    NativeOpExcutioner::execTransformAny(transform::IsMax, x.buffer(), x.shapeInfo(), z.buffer(), z.shapeInfo(), extra, nullptr, nullptr);

    // z.printIndexedBuffer("z");
 for (int e = 0; e < z.lengthOf(); e++) {
     if (e >= z.lengthOf() / 2)
         ASSERT_TRUE(z.e<bool>(e));
     else
         ASSERT_FALSE(z.e<bool>(e));
 }
}

TEST_F(LegacyOpsTests, BroadcastingTests_1) {
    auto x = NDArrayFactory::create<double>('c', {5, 5});
    x.assign(0.0f);

    auto row = NDArrayFactory::create<double>('c', {1, 5});
    row.linspace(1);
    auto axis = NDArrayFactory::create<int>('c', {1}, {1});
    nd4j::ops::LegacyBroadcastOp op(broadcast::Add);
    Nd4jStatus status = op.execute({&x, &row, &axis}, {&x}, {}, {}, {});

    ASSERT_EQ(ND4J_STATUS_OK, status);

    auto list = x.allTensorsAlongDimension({1});
    x.printIndexedBuffer("Output broadcast");
    list->at(0)->printIndexedBuffer("Column 0:");
    for (int e = 0; e < list->size(); e++)
        ASSERT_TRUE(row.equalsTo(list->at(e)));

    delete list;
}

TEST_F(LegacyOpsTests, BroadcastingTests_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 1, 1, 1, 1});
    auto y = NDArrayFactory::create<double>('c', {10, 5});
    auto e = NDArrayFactory::create<double>('c', {10, 5});
    y.assign(3.0);
    e.assign(4.0);

    int axis = 1;
    shape::TAD tad;
    tad.init(y.shapeInfo(), &axis, 1);
    tad.createTadOnlyShapeInfo();
    tad.createOffsets();

    shape::printShapeInfoLinear("tad shape", tad.tadOnlyShapeInfo);

    NativeOpExcutioner::execInverseBroadcast(broadcast::Add, x.buffer(), x.shapeInfo(), y.buffer(), y.shapeInfo(), y.buffer(), y.shapeInfo(), &axis, 1, tad.tadOnlyShapeInfo, tad.tadOffsets, tad.tadOnlyShapeInfo, tad.tadOffsets);

    ASSERT_EQ(e, y);
}

TEST_F(LegacyOpsTests, PowDerivative_1) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto exp = NDArrayFactory::create<float>('c', {5, 5});
    x.assign(3.f);
    exp.assign(6.f);

    float p = 2.0f;

    x.applyScalar(scalar::PowDerivative, p);

    ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, reduce3_1) {
    
    Nd4jLong yShape[2] = {4,4};
    Nd4jLong xShape[1] = {4};
    float y[16] ={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float x[4] = {1,2,3,4};
    int dimension[1] = {1};
    int dimensionLength = 1;
    int opNum = 1;
    float extraVals[1] = {0};
    float result[4] = {0.0,0.0,0.0,0.0};

    std::vector<int> dim = {1};

    auto shapeBuffer  = nd4j::ShapeBuilders::createShapeInfo(nd4j::DataType::FLOAT32, 'c', 2, yShape);
    auto xShapeBuffer = nd4j::ShapeBuilders::createShapeInfo(nd4j::DataType::FLOAT32, 'c', 1, xShape);    

    //int *tadShapeBuffer = shape::computeResultShape(shapeBuffer,dimension,dimensionLength);
    auto tadShapeBuffer = nd4j::ShapeUtils::evalReduceShapeInfo('c', dim, shapeBuffer, false, true, nullptr);
    functions::reduce3::Reduce3<float, float>::exec(opNum, x, xShapeBuffer, extraVals, y, shapeBuffer, result, tadShapeBuffer, dimension, dimensionLength);

    float distancesAssertion[4] = {0.0,8.0,16.0,24.0};
    for(int i = 0; i < 4; i++) 
        ASSERT_EQ(distancesAssertion[i],result[i]);
    
    delete[] shapeBuffer;
    delete[] tadShapeBuffer;
    delete[] xShapeBuffer;
}


TEST_F(LegacyOpsTests, Reduce3_2) {
    auto x = NDArrayFactory::create<float>('c', {5, 5});
    auto y = NDArrayFactory::create<float>('c', {5});
    auto z = NDArrayFactory::create<float>('c', {5});

    auto dim = NDArrayFactory::create<int>('c', {1}, {1});

    NativeOps nativeOps;
    nativeOps.execReduce3(nullptr, reduce3::CosineSimilarity, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), dim.buffer(), dim.shapeInfo(), dim.specialBuffer(), dim.specialShapeInfo(),
                          nullptr, nullptr, nullptr, nullptr);
}

TEST_F(LegacyOpsTests, Reduce3_3) {
    auto x = NDArrayFactory::create<double>('c', {3, 5}, {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                                                          -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                                                          0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247});

    auto y = NDArrayFactory::create<double>('c', {5}, {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673});
    auto e = NDArrayFactory::create<double>('c', {3}, {0.577452, 0.0, 1.80182});
    auto z = NDArrayFactory::create<double>('c', {3});

    auto dim = NDArrayFactory::create<int>('c', {1}, {1});

    NativeOps nativeOps;

    nativeOps.execReduce3(nullptr, reduce3::CosineDistance, 
                        x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), 
                        nullptr, 
                        y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), 
                        z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), 
                        dim.buffer(), dim.shapeInfo(), dim.specialBuffer(), dim.specialShapeInfo(),
                          nullptr, nullptr, nullptr, nullptr);

    // z.printIndexedBuffer("z");

    ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, Reduce3_4) {
    auto x = NDArrayFactory::create<double>('c', {3, 5}, {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                                                          -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                                                          0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247});

    auto y = NDArrayFactory::create<double>('c', {1, 5}, {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673});
    auto e = NDArrayFactory::create<double>('c', {1, 3}, {0.577452, 0.0, 1.80182});
    auto z = NDArrayFactory::create<double>('c', {1, 3});

    auto dim = NDArrayFactory::create<int>('c', {1}, {1});

    NativeOps nativeOps;

    nativeOps.execReduce3(nullptr, reduce3::CosineDistance,
                          x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                          nullptr,
                          y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                          z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                          dim.buffer(), dim.shapeInfo(), dim.specialBuffer(), dim.specialShapeInfo(),
                          nullptr, nullptr, nullptr, nullptr);

    // z.printIndexedBuffer("z");

    ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, Reduce3_5) {
    auto x = NDArrayFactory::create<double>('c', {3, 5}, {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                                                          -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                                                          0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247});

    auto y = NDArrayFactory::create<double>('c', {1, 5}, {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673});
    auto e = NDArrayFactory::create<double>('c', {1, 3}, {0.577452, 0.0, 1.80182});
    auto z = NDArrayFactory::create<double>('c', {1, 3});

    auto dim = NDArrayFactory::create<int>('c', {1}, {1});

    NativeOps nativeOps;

    nativeOps.execReduce3(nullptr, reduce3::CosineDistance,
                          x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                          nullptr,
                          y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                          z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                          dim.buffer(), dim.shapeInfo(), dim.specialBuffer(), dim.specialShapeInfo(),
                          nullptr, nullptr, nullptr, nullptr);

    z.printIndexedBuffer("z");

    ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, test_Reduce3_All_1) {
    auto x = NDArrayFactory::create<float>('c', {1000, 100});
    auto y = NDArrayFactory::create<float>('c', {1, 100});
    auto z = NDArrayFactory::create<float>('c', {1000, 1});
    auto dim = NDArrayFactory::create<int>('c', {1}, {-1});

    auto tadPackX = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(x.shapeInfo(), -1);
    auto tadPackY = nd4j::ConstantTadHelper::getInstance()->tadForDimensions(y.shapeInfo(), -1);

    NativeOps ops;
    ops.execReduce3All(nullptr, reduce3::EuclideanDistance, x.buffer(), x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(),
                       nullptr, y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(),
                       z.buffer(), z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                       dim.buffer(), dim.shapeInfo(), dim.specialBuffer(), dim.specialShapeInfo(),
                       tadPackX.primaryShapeInfo(), tadPackX.primaryOffsets(),
                       tadPackY.primaryShapeInfo(), tadPackY.primaryOffsets());
}

TEST_F(LegacyOpsTests, Softmax_119_1) {
    auto x = NDArrayFactory::create<float>('c', {10, 10});
    x.linspace(1.0);

    x.applyTransform(transform::StrictOps::SoftMax);
}

TEST_F(LegacyOpsTests, Softmax_119_2) {
    auto x = NDArrayFactory::create<float>('f', {10, 5});
    auto z = NDArrayFactory::create<float>('f', {10, 5});
    x.linspace(1.0);

    x.applyTransform(transform::StrictOps::SoftMax, &z);
}