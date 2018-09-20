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
// @author raver119@gmail.com
//

#include <helpers/shape.h>
#include "testlayers.h"
#include <ops/declarable/headers/shape.h>

using namespace nd4j;
using namespace nd4j::graph;

class ShapeTests : public testing::Test {
public:

};


TEST_F(ShapeTests, Test_Basics_1) {
    Nd4jLong shape[] = {2, 5, 3, 3, 1, 0, 1, 99};

    ASSERT_EQ(2, shape::rank(shape));
    ASSERT_EQ(1, shape::elementWiseStride(shape));
    ASSERT_EQ(5, shape::sizeAt(shape, 0));
    ASSERT_EQ(3, shape::sizeAt(shape, 1));
    ASSERT_EQ('c', shape::order(shape));
}


TEST_F(ShapeTests, Test_Basics_2) {
    Nd4jLong shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};

    ASSERT_EQ(4, shape::rank(shape));
    ASSERT_EQ(-1, shape::elementWiseStride(shape));
    ASSERT_EQ(2, shape::sizeAt(shape, 0));
    ASSERT_EQ(3, shape::sizeAt(shape, 1));
    ASSERT_EQ(4, shape::sizeAt(shape, 2));
    ASSERT_EQ(5, shape::sizeAt(shape, 3));
    ASSERT_EQ('f', shape::order(shape));
}


TEST_F(ShapeTests, Test_tadLength_1) {
    Nd4jLong shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    int axis[] = {2, 3};

    ASSERT_EQ(20, shape::tadLength(shape, axis, 2));
}


TEST_F(ShapeTests, Test_ShapeEquality_1) {
    Nd4jLong shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    Nd4jLong shape_GOOD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, 1, 99};
    Nd4jLong shape_BAD[] = {4, 3, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    

    ASSERT_TRUE(shape::equalsSoft(shape, shape_GOOD));
    ASSERT_FALSE(shape::equalsSoft(shape, shape_BAD));
}

TEST_F(ShapeTests, Test_ShapeEquality_2) {
    Nd4jLong shape[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    Nd4jLong shape_GOOD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 102};
    Nd4jLong shape_BAD[] = {4, 2, 3, 4, 5, 60, 20, 5, 1, 0, -1, 99};
    

    ASSERT_TRUE(shape::equalsStrict(shape, shape_GOOD));
    ASSERT_FALSE(shape::equalsStrict(shape, shape_BAD));
}

TEST_F(ShapeTests, Test_Ind2SubC_1) {
    Nd4jLong shape[] = {3, 5};
    auto c0 = shape::ind2subC(2, shape, 0);

    ASSERT_EQ(0, c0[0]);
    ASSERT_EQ(0, c0[1]);

    auto c1 = shape::ind2subC(2, shape, 1);

    ASSERT_EQ(0, c1[0]);
    ASSERT_EQ(1, c1[1]);

    auto c6 = shape::ind2subC(2, shape, 5);

    ASSERT_EQ(1, c6[0]);
    ASSERT_EQ(0, c6[1]);

    delete[] c0;
    delete[] c1;
    delete[] c6;
}

TEST_F(ShapeTests, Test_Ind2Sub_1) {
    Nd4jLong shape[] = {3, 5};
    auto c0 = shape::ind2sub(2, shape, 0);

    ASSERT_EQ(0, c0[0]);
    ASSERT_EQ(0, c0[1]);

    auto c1 = shape::ind2sub(2, shape, 1);

    ASSERT_EQ(1, c1[0]);
    ASSERT_EQ(0, c1[1]);

    auto c6 = shape::ind2sub(2, shape, 5);

    ASSERT_EQ(2, c6[0]);
    ASSERT_EQ(1, c6[1]);

    delete[] c0;
    delete[] c1;
    delete[] c6;
}

TEST_F(ShapeTests, Test_ShapeDetector_1) {
    Nd4jLong shape[] = {2, 5, 3, 3, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_2) {
    Nd4jLong shape[] = {3, 2, 5, 3, 15, 3, 1, 0, 1, 99};

    ASSERT_FALSE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_3) {
    Nd4jLong shape[] = {2, 1, 3, 3, 1, 0, 1, 99};

    ASSERT_FALSE(shape::isColumnVector(shape));
    ASSERT_TRUE(shape::isVector(shape));
    ASSERT_TRUE(shape::isRowVector(shape));
    ASSERT_FALSE(shape::isMatrix(shape));
}


TEST_F(ShapeTests, Test_ShapeDetector_4) {
    Nd4jLong shape[] = {2, 3, 1, 1, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isColumnVector(shape));
    ASSERT_TRUE(shape::isVector(shape));
    ASSERT_FALSE(shape::isRowVector(shape));
    ASSERT_FALSE(shape::isMatrix(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_5) {
    Nd4jLong shape[] = {2, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_TRUE(shape::isScalar(shape));
    ASSERT_FALSE(shape::isMatrix(shape));

    // edge case here. Technicaly it's still a vector with length of 1
    ASSERT_TRUE(shape::isVector(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_6) {
    Nd4jLong shape[] = {2, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_EQ(8, shape::shapeInfoLength(shape));
    ASSERT_EQ(64, shape::shapeInfoByteLength(shape));
}

TEST_F(ShapeTests, Test_ShapeDetector_7) {
    Nd4jLong shape[] = {3, 1, 1, 1, 1, 1, 1, 0, 1, 99};

    ASSERT_EQ(10, shape::shapeInfoLength(shape));
    ASSERT_EQ(80, shape::shapeInfoByteLength(shape));
}

TEST_F(ShapeTests, Test_Transpose_1) {
    Nd4jLong shape[] = {3, 2, 5, 3, 15, 3, 1, 0, 1, 99};
    Nd4jLong exp[] = {3, 3, 5, 2, 1, 3, 15, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}

TEST_F(ShapeTests, Test_Transpose_2) {
    Nd4jLong shape[] = {2, 5, 3, 3, 1, 0, 1, 99};
    Nd4jLong exp[] = {2, 3, 5, 1, 3, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}

TEST_F(ShapeTests, Test_Transpose_3) {
    Nd4jLong shape[] = {2, 1, 3, 3, 1, 0, 1, 99};
    Nd4jLong exp[] = {2, 3, 1, 1, 3, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}


TEST_F(ShapeTests, Test_Transpose_4) {
    Nd4jLong shape[] = {4, 2, 3, 4, 5, 5, 4, 3, 2, 0, 1, 99};
    Nd4jLong exp[] = {4, 5, 4, 3, 2, 2, 3, 4, 5, 0, 1, 102};

    shape::transposeInplace(shape);

    ASSERT_TRUE(shape::equalsStrict(exp, shape));
}

TEST_F(ShapeTests, Test_Edge_1) {
    auto x = NDArrayFactory::_create<float>('f', {1, 4, 1, 4});
    x.linspace(1);

    x.reshapei('c', {4, 4});

    //x.printShapeInfo("reshape0");
    //x.printIndexedBuffer("x i");
    //x.printBuffer("x r");

    x.reshapei({4, 1, 1, 4});

    //x.printShapeInfo("reshape1");
}

TEST_F(ShapeTests, Test_Edge_2) {
    auto x = NDArrayFactory::_create<float>('c', {1, 4, 1, 3});

    x.reshapei('f', {3, 4});

    //x.printShapeInfo("reshape0");

    x.reshapei({3, 1, 1, 4});

    //x.printShapeInfo("reshape1");
}


TEST_F(ShapeTests, Test_Remove_Index_1) {
    int array[] = {1, 2, 3};
    int idx[] = {0};
    int result[2];
    shape::removeIndex(array, idx, 3, 1, result);

    ASSERT_EQ(2, result[0]);
    ASSERT_EQ(3, result[1]);
}

TEST_F(ShapeTests, Test_Remove_Index_2) {
    int array[] = {1, 2, 3};
    int idx[] = {1};
    int result[2];
    shape::removeIndex(array, idx, 3, 1, result);

    ASSERT_EQ(1, result[0]);
    ASSERT_EQ(3, result[1]);
}

TEST_F(ShapeTests, Test_Remove_Index_3) {
    int array[] = {1, 2, 3};
    int idx[] = {2};
    int result[2];
    shape::removeIndex(array, idx, 3, 1, result);

    ASSERT_EQ(1, result[0]);
    ASSERT_EQ(2, result[1]);
}

TEST_F(ShapeTests, Test_Remove_Index_4) {
    int array[] = {1, 2, 3};
    int idx[] = {0, 2};
    int result[1];
    shape::removeIndex(array, idx, 3, 2, result);

    ASSERT_EQ(2, result[0]);
}

TEST_F(ShapeTests, Test_Remove_Index_5) {
    int array[] = {1, 2, 3};
    int idx[] = {1, 0};
    int result[1];
    shape::removeIndex(array, idx, 3, 2, result);

    ASSERT_EQ(3, result[0]);
}

TEST_F(ShapeTests, Test_Remove_Index_6) {
    int array[] = {1, 2, 3};
    int idx[] = {1, 2};
    int result[1];
    shape::removeIndex(array, idx, 3, 2, result);

    ASSERT_EQ(1, result[0]);
}

TEST_F(ShapeTests, Tests_Transpose_119_1) {
    auto x = NDArrayFactory::_create<float>('c', {3, 2});
    auto y = NDArrayFactory::_create<float>('c', {2}, {1.0f, 0.0f});
    auto z = NDArrayFactory::_create<float>('c', {2, 3});

    x.linspace(1.f);

    auto e = x.permute({1, 0});
    e->streamline('c');

    nd4j::ops::transpose op;
    auto result = op.execute({&x, &y}, {&z}, {}, {});

    ASSERT_EQ(Status::OK(), result);
    ASSERT_TRUE(e->isSameShape(z));
    ASSERT_TRUE(e->equalsTo(z));

    delete e;
}

TEST_F(ShapeTests, Tests_Transpose_119_2) {
    auto x = NDArrayFactory::_create<float>('c', {3, 5});
    x.linspace(1.f);

    auto exp = x.transpose();

    nd4j::ops::transpose op;
    auto result = op.execute({&x},{}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete exp;
    delete result;
}

TEST_F(ShapeTests, Tests_Transpose_119_3) {
    auto x = NDArrayFactory::_create<float>('c', {3, 5});
    x.linspace(1.f);

    auto z = NDArrayFactory::_create<float>('c', {5, 3});

    auto exp = x.transpose();

    nd4j::ops::transpose op;
    auto result = op.execute({&x}, {&z}, {}, {});
    ASSERT_EQ(Status::OK(), result);

    ASSERT_TRUE(exp->isSameShape(z));
    ASSERT_TRUE(exp->equalsTo(z));

    delete exp;
}