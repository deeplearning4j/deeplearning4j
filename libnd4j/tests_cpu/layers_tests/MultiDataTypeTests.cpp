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

#include "testlayers.h"
#include <array/ArrayOptions.h>
#include <NDArray.h>
#include <NDArrayFactory.h>


using namespace nd4j;

class MultiDataTypeTests : public testing::Test {
public:

};

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_1) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::INT32, nd4j::FLOAT32);

    ASSERT_EQ(nd4j::FLOAT32, dtype);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_2) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::INT32, nd4j::DOUBLE);
    ASSERT_EQ(nd4j::DOUBLE, dtype);

    ASSERT_EQ(nd4j::DOUBLE, DataTypeUtils::pickPairwiseResultType(nd4j::DOUBLE, nd4j::INT32));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_3) {
    auto dtype = DataTypeUtils::pickPairwiseResultType(nd4j::FLOAT32, nd4j::DOUBLE);
    ASSERT_EQ(nd4j::FLOAT32, dtype);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<double>('c', {2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x + y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_2) {
    auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<double>(2.0);
    auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_3) {
    auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<double>(2.0);
    auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_4) {
    auto x = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    auto y = NDArrayFactory::create<float>(2.0);
    auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_5) {
    auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<Nd4jLong>(2);
    auto e = NDArrayFactory::create<int>('c', {2, 3}, {0, 2, 4, 6, 8, 10});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_6) {
    auto x = NDArrayFactory::create<Nd4jLong>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
    auto y = NDArrayFactory::create<int>(2);
    auto e = NDArrayFactory::create<Nd4jLong >('c', {2, 3}, {0, 2, 4, 6, 8, 10});

    auto z = x * y;

    ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test1) {
    NDArray x('c', {2, 3}, {0, 1, 2, 3, 4, 5}, nd4j::DataType::UINT8);
    NDArray exp('c', {2, 3}, {10, 10, 10, 10, 10, 10}, nd4j::DataType::UINT8);
    
    const double number = 10.8;
    x = number;
    
    ASSERT_EQ(x,exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test2) {
    NDArray x('c', {2, 3}, {0, 1, 2, 3, 4, 5}, nd4j::DataType::INT64);
    NDArray exp('c', {2, 3}, {1, 1, 1, 1, 1, 1}, nd4j::DataType::INT64);
    
    const bool number = 1000;
    x = number;
    
    ASSERT_EQ(x,exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test3) {
    NDArray x('c', {2, 3}, {0, 1, 0, 1, 0, 1}, nd4j::DataType::BOOL);
    NDArray exp('c', {2, 3}, {1, 1, 1, 1, 1, 1}, nd4j::DataType::BOOL);
    
    const int number = 1000;
    x = number;
    
    ASSERT_EQ(x,exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_repeat_test1) {
    NDArray x('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, nd4j::DataType::HALF);
    NDArray y('c', {2, 4}, nd4j::DataType::UINT8);
    NDArray exp('c', {2, 4}, {0, 0, 1, 1, 2, 2, 3, 3}, nd4j::DataType::UINT8);
    
    x.repeat(1, y);
    
    ASSERT_EQ(y, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_bufferAsT_test1) {
    NDArray x('f', {2}, {1.5, 3.5}, nd4j::DataType::FLOAT32);
    NDArray y('c', {0}, {1.5}, nd4j::DataType::FLOAT32);
    
    const int* buffX = x.bufferAsT<int>();
    const int* buffY = y.bufferAsT<int>();

    ASSERT_EQ(*buffX, *buffY);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_test1) {
    NDArray x('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::UINT8);
    NDArray exp('c', {2,2}, {10, 10, 20, 20}, nd4j::DataType::UINT8);
    
    NDArray scalar1('c', {0}, {10.5}, nd4j::DataType::FLOAT32);
    NDArray scalar2('c', {0}, {20.8}, nd4j::DataType::DOUBLE);    
    
    x(0,{0}).assign(scalar1);
    x(1,{0}).assign(scalar2);

    ASSERT_EQ(x, exp);

    x.assign(exp);

    ASSERT_EQ(x, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test1) {
    NDArray x('f', {2,2}, {0, 1.5, 2.5, 3.5}, nd4j::DataType::HALF);
    NDArray exp1('c', {0}, {3}, nd4j::DataType::INT64);
    NDArray exp2('c', {1,1}, {1}, nd4j::DataType::INT64);
    NDArray exp3('c', {2}, {1,2}, nd4j::DataType::INT64);

    auto* scalar1 = x.reduceAlongDimension(nd4j::reduce::CountNonZero, {}/*whole range*/);
    ASSERT_EQ(*scalar1, exp1);

    auto* scalar2 = x.reduceAlongDimension(nd4j::reduce::CountZero, {}/*whole range*/, true);
    ASSERT_EQ(*scalar2, exp2);

    auto* scalar3 = x.reduceAlongDimension(nd4j::reduce::CountNonZero, {1});
    ASSERT_EQ(*scalar3, exp3);

    delete scalar1;
    delete scalar2;
    delete scalar3;
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test2) {
    NDArray x('c', {2, 2}, {0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray exp1('c', {0}, {1.5}, nd4j::DataType::FLOAT32);
    NDArray exp2('c', {2}, {0.5,2.5}, nd4j::DataType::FLOAT32);
    
    auto* scalar1 = x.reduceAlongDimension(nd4j::reduce::Mean, {}/*whole range*/);
    // scalar1->printShapeInfo();
    // scalar1->printIndexedBuffer();
    ASSERT_EQ(*scalar1, exp1);

    auto* scalar2 = x.reduceAlongDimension(nd4j::reduce::Mean, {1});
    ASSERT_EQ(*scalar2, exp2);

    delete scalar1;
    delete scalar2;    
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test3) {
    NDArray x('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, nd4j::DataType::HALF);
    NDArray exp1('c', {0}, {8.}, nd4j::DataType::HALF);
    NDArray exp2('c', {2}, {2.,6.}, nd4j::DataType::HALF);
    
    auto scalar1 = x.reduceAlongDims(nd4j::reduce::Sum, {}/*whole range*/);
    ASSERT_EQ(scalar1, exp1);

    auto scalar2 = x.reduceAlongDims(nd4j::reduce::Sum, {1});
    ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test4) {
    NDArray x('c', {2, 2}, {10.5, 1.5, -2.5, -3.5}, nd4j::DataType::HALF);
    NDArray exp1('c', {0}, {1}, nd4j::DataType::BOOL);
    NDArray exp2('c', {2}, {1,0}, nd4j::DataType::BOOL);
    
    auto scalar1 = x.reduceAlongDims(nd4j::reduce::IsPositive, {}/*whole range*/);
    ASSERT_EQ(scalar1, exp1);

    auto scalar2 = x.reduceAlongDims(nd4j::reduce::IsPositive, {1});
    ASSERT_EQ(scalar2, exp2);
}
