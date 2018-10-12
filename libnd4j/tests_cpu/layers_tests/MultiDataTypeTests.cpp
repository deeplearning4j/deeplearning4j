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

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_varianceNumber_test1) {
    NDArray x('f', {2, 2}, {0, 1, 2, 3}, nd4j::DataType::INT64);
    NDArray exp1('c', {0}, {1.666666667}, nd4j::DataType::FLOAT32);
    NDArray exp2('c', {0}, {1.118033989}, nd4j::DataType::FLOAT32);
    
    auto scalar1 = x.varianceNumber(variance::SummaryStatsVariance);
    ASSERT_EQ(scalar1, exp1);

    auto scalar2 = x.varianceNumber(variance::SummaryStatsStandardDeviation, false);
    ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlus_test1) {
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, nd4j::DataType::FLOAT32);
    NDArray x3('c', {2}, {-1, -2},            nd4j::DataType::FLOAT32);

    NDArray exp('c', {2, 2}, {-1, -1, 1, 1},  nd4j::DataType::FLOAT32);
        
    ASSERT_EQ(x1+x2, exp);
    ASSERT_EQ(x1+x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlus_test2) {
    
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::FLOAT32);
    NDArray x3('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::HALF);
    const double val1 = -2;
    const int val2 = -2;
    NDArray exp1('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::FLOAT32);
    NDArray exp3('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::HALF);
    
    ASSERT_EQ(x1+val1, exp1);
    ASSERT_EQ(val1+x1, exp1);

    ASSERT_EQ(x2+val2, exp2);
    ASSERT_EQ(val2+x2, exp2);    

    ASSERT_EQ(x3+val1, exp3);
    ASSERT_EQ(val1+x3, exp3);    
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinus_test1) {
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, nd4j::DataType::HALF);
    NDArray x3('c', {2}, {-1, -2},            nd4j::DataType::HALF);

    NDArray exp('c', {2, 2}, {1, 3, 3, 5},  nd4j::DataType::HALF);
        
    ASSERT_EQ(x1-x2, exp);
    ASSERT_EQ(x1-x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinus_test2) {
    
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::FLOAT32);
    NDArray x3('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::HALF);
    const double val1 = 2;
    const int val2 = 2;
    NDArray exp1('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::DOUBLE);    
    NDArray exp2('c', {2,2}, {2, 1, 0, -1},   nd4j::DataType::DOUBLE);
    NDArray exp3('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::FLOAT32);
    NDArray exp4('c', {2,2}, {-2, -1, 0, 1},  nd4j::DataType::HALF);
    NDArray exp5('c', {2,2}, {2, 1, 0, -1},   nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {2, 1, 0, -1},   nd4j::DataType::HALF);
    
    ASSERT_EQ(x1-val1, exp1);
    ASSERT_EQ(val1-x1, exp2);

    ASSERT_EQ(x2-val2, exp3);
    ASSERT_EQ(val2-x2, exp5);

    ASSERT_EQ(x3-val1, exp4);
    ASSERT_EQ(val1-x3, exp6);
}

//////////////////////////////////////////////////////////////////////////////// multiply 
TEST_F(MultiDataTypeTests, ndarray_operatorMultiply_test1) {
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, nd4j::DataType::DOUBLE);
    NDArray x3('c', {2}, {-1, -2},            nd4j::DataType::DOUBLE);

    NDArray exp('c', {2, 2}, {0, -2, -2, -6},  nd4j::DataType::DOUBLE);
        
    ASSERT_EQ(x1*x2, exp);
    ASSERT_EQ(x1*x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMultiply_test2) {
    
    NDArray x1('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::INT64);
    NDArray x2('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::FLOAT32);
    NDArray x3('c', {2, 2}, {0, 1, 2, 3},     nd4j::DataType::HALF);
    const double val1 = -2;
    const int val2 = -2;
    NDArray exp1('c', {2,2}, {0, -2, -4, -6},  nd4j::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {0, -2, -4, -6},  nd4j::DataType::FLOAT32);
    NDArray exp3('c', {2,2}, {0, -2, -4, -6},  nd4j::DataType::HALF);
    
    ASSERT_EQ(x1*val1, exp1);
    ASSERT_EQ(val1*x1, exp1);

    ASSERT_EQ(x2*val2, exp2);
    ASSERT_EQ(val2*x2, exp2);    

    ASSERT_EQ(x3*val1, exp3);
    ASSERT_EQ(val1*x3, exp3);    
}


//////////////////////////////////////////////////////////////////////////////// multiply 
TEST_F(MultiDataTypeTests, ndarray_operatorDivide_test1) {
    NDArray x1('c', {2, 2}, {4, 1, 2, 3},     nd4j::DataType::HALF);
    NDArray x2('c', {2, 2}, {-1, -2, -1, -9}, nd4j::DataType::DOUBLE);
    NDArray x3('c', {2}, {-1, -2},            nd4j::DataType::FLOAT32);

    NDArray exp1('c', {2, 2}, {-4, -0.5, -2, -0.3333333},  nd4j::DataType::HALF);
    NDArray exp2('c', {2, 2}, {-0.25, -2, -0.5, -0.666667},  nd4j::DataType::HALF);
        
    ASSERT_EQ(x1/x2, exp1);    
    ASSERT_EQ(x3/x1, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorDivide_test2) {
    
    NDArray x1('c', {2, 2}, {1, 2, 3, 4},     nd4j::DataType::INT64);    
    NDArray x2('c', {2, 2}, {1, 2, 3, 4},     nd4j::DataType::FLOAT32);
    NDArray x3('c', {2, 2}, {1, 2, 3, 4},     nd4j::DataType::HALF);
    const double val1 = 2;
    const int val2 = -2;
    NDArray exp1('c', {2,2}, {0.5, 1, 1.5, 2},  nd4j::DataType::DOUBLE);
    NDArray exp2('c', {2,2}, {2, 1, 0.666667, 0.5},   nd4j::DataType::DOUBLE);
    NDArray exp3('c', {2,2}, {0, -1, -1, -2},  nd4j::DataType::INT64);
    NDArray exp4('c', {2,2}, {-2, -1, 0., 0.},   nd4j::DataType::INT64);    
    NDArray exp5('c', {2,2}, {-0.5, -1, -1.5, -2},  nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {-2, -1, -0.666667, -0.5},  nd4j::DataType::FLOAT32);    
    NDArray exp7('c', {2,2}, {0.5, 1, 1.5, 2},  nd4j::DataType::HALF);
    NDArray exp8('c', {2,2}, {2, 1, 0.666667, 0.5},   nd4j::DataType::HALF);
    
    ASSERT_EQ(x1/val1, exp1);
    ASSERT_EQ(val1/x1, exp2);

    ASSERT_EQ(x1/val2, exp3);
    ASSERT_EQ(val2/x1, exp4);

    ASSERT_EQ(x2/val2, exp5);
    ASSERT_EQ(val2/x2, exp6);

    ASSERT_EQ(x3/val1, exp7);
    ASSERT_EQ(val1/x3, exp8);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlusEqual_test1) {
    
    NDArray scalar1('c', {0}, {4}, nd4j::DataType::INT32);
    NDArray scalar2('c', {0}, {1.5}, nd4j::DataType::HALF);
    
    NDArray x1('c', {2,3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5},  nd4j::DataType::FLOAT32);
    NDArray x2('c', {3,2}, {10, 20, 30, 40, 50, 60},  nd4j::DataType::INT64);
    NDArray x3('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::INT64);
    NDArray x4('c', {2},   {0.4, 0.5},  nd4j::DataType::HALF);
    NDArray x5('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::HALF);
    NDArray x6('c', {2},   {0.4, 0.5},  nd4j::DataType::FLOAT32);

    NDArray exp1('c', {0}, {5},  nd4j::DataType::INT32);
    NDArray exp2('c', {0}, {6.5},  nd4j::DataType::HALF);
    NDArray exp3('c', {3,2}, {11, 22, 33, 44, 55, 66},  nd4j::DataType::INT64);
    NDArray exp4('c', {2,3}, {12.5, 24.5, 36.5, 48.5, 60.5, 72.5},  nd4j::DataType::FLOAT32);
    NDArray exp5('c', {2,2}, {0.4, 1.5, 2.4, 3.5},  nd4j::DataType::HALF);
    
    scalar1 += scalar2;
    ASSERT_EQ(scalar1, exp1);

    scalar2 += scalar1;
    ASSERT_EQ(scalar2, exp2);

    x2 += x1;
    ASSERT_EQ(x2, exp3);

    x1 += x2;
    ASSERT_EQ(x1, exp4);

    x4 += x3;
    ASSERT_EQ(x4, exp5);

    x6 += x5;
    ASSERT_EQ(x6, exp5);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlusEqual_test2) {
    
    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);    

    const Nd4jLong val1 = 1;
    const float16  val2 = 1.5;
    const double   val3 = 2.2;

    NDArray exp1('c', {2,2}, {1, 2, 3, 4},  nd4j::DataType::FLOAT32);
    NDArray exp2('c', {2,2}, {1, 2, 3, 4},  nd4j::DataType::INT32);
    NDArray exp3('c', {2,2}, {2.5, 3.5, 4.5, 5.5}, nd4j::DataType::FLOAT32);
    NDArray exp4('c', {2,2}, {2, 3, 4.5, 5}, nd4j::DataType::INT32);
    NDArray exp5('c', {2,2}, {4.7, 5.7, 6.7, 7.7}, nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {4, 5, 6, 7}, nd4j::DataType::INT32);
    
    x1 += val1;
    ASSERT_EQ(x1, exp1);

    x2 += val1;
    ASSERT_EQ(x2, exp2);

    x1 += val2;
    ASSERT_EQ(x1, exp3);

    x2 += val2;
    ASSERT_EQ(x2, exp4);

    x1 += val3;
    ASSERT_EQ(x1, exp5);

    x2 += val3;
    ASSERT_EQ(x2, exp6);    
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinusEqual_test1) {
    
    NDArray scalar1('c', {0}, {4}, nd4j::DataType::INT32);
    NDArray scalar2('c', {0}, {1.5}, nd4j::DataType::HALF);
    
    NDArray x1('c', {2,3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5},  nd4j::DataType::FLOAT32);
    NDArray x2('c', {3,2}, {10, 20, 30, 40, 50, 60},  nd4j::DataType::INT64);
    NDArray x3('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::INT64);
    NDArray x4('c', {2},   {0.4, 0.5},  nd4j::DataType::HALF);
    NDArray x5('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::HALF);
    NDArray x6('c', {2},   {0.4, 0.5},  nd4j::DataType::FLOAT32);

    NDArray exp1('c', {0}, {2},  nd4j::DataType::INT32);
    NDArray exp2('c', {0}, {-0.5},  nd4j::DataType::HALF);
    NDArray exp3('c', {3,2}, {8, 17, 26, 35, 44, 53},  nd4j::DataType::INT64);
    NDArray exp4('c', {2,3}, {-6.5, -14.5, -22.5, -30.5, -38.5, -46.5},  nd4j::DataType::FLOAT32);
    NDArray exp5('c', {2,2}, {0.4, -0.5, -1.6, -2.5},  nd4j::DataType::HALF);
        
    scalar1 -= scalar2;
    ASSERT_EQ(scalar1, exp1);

    scalar2 -= scalar1;
    ASSERT_EQ(scalar2, exp2);

    x2 -= x1;
    ASSERT_EQ(x2, exp3);

    x1 -= x2;
    ASSERT_EQ(x1, exp4);

    x4 -= x3;
    ASSERT_EQ(x4, exp5);

    x6 -= x5;
    ASSERT_EQ(x6, exp5);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinusEqual_test2) {
    
    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);    

    const Nd4jLong val1 = 1;
    const float16  val2 = 1.5;
    const double   val3 = 2.2;

    NDArray exp1('c', {2,2}, {-1, 0, 1, 2},  nd4j::DataType::FLOAT32);
    NDArray exp2('c', {2,2}, {-1, 0, 1, 2},  nd4j::DataType::INT32);
    NDArray exp3('c', {2,2}, {-2.5, -1.5, -0.5, 0.5}, nd4j::DataType::FLOAT32);
    NDArray exp4('c', {2,2}, {-2., -1., 0., 0.}, nd4j::DataType::INT32);
    NDArray exp5('c', {2,2}, {-4.7, -3.7, -2.7, -1.7}, nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {-4, -3, -2, -2}, nd4j::DataType::INT32);
    
    x1 -= val1;
    ASSERT_EQ(x1, exp1);

    x2 -= val1;
    ASSERT_EQ(x2, exp2);

    x1 -= val2;
    ASSERT_EQ(x1, exp3);

    x2 -= val2;    
    ASSERT_EQ(x2, exp4);

    x1 -= val3;
    ASSERT_EQ(x1, exp5);

    x2 -= val3;    
    ASSERT_EQ(x2, exp6);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMultiplyEqual_test1) {
    
    NDArray scalar1('c', {0}, {3}, nd4j::DataType::INT32);
    NDArray scalar2('c', {0}, {2.5}, nd4j::DataType::HALF);
    
    NDArray x1('c', {2,3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5},  nd4j::DataType::FLOAT32);
    NDArray x2('c', {3,2}, {1, 2, 3, 4, 5, 6},  nd4j::DataType::INT64);
    NDArray x3('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::INT64);
    NDArray x4('c', {2},   {0.4, 0.5},  nd4j::DataType::HALF);
    NDArray x5('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::HALF);
    NDArray x6('c', {2},   {0.4, 0.5},  nd4j::DataType::FLOAT32);

    NDArray exp1('c', {0}, {7},  nd4j::DataType::INT32);
    NDArray exp2('c', {0}, {17.5},  nd4j::DataType::HALF);
    NDArray exp3('c', {3,2}, {1, 5, 10, 18, 27, 39},  nd4j::DataType::INT64);
    NDArray exp4('c', {2,3}, {1.5, 12.5, 35, 81, 148.5, 253.5},  nd4j::DataType::FLOAT32);
    NDArray exp5('c', {2,2}, {0., 0.5, 0.8, 1.5},  nd4j::DataType::HALF);
    
    scalar1 *= scalar2;
    ASSERT_EQ(scalar1, exp1);

    scalar2 *= scalar1;    
    ASSERT_EQ(scalar2, exp2);

    x2 *= x1;
    ASSERT_EQ(x2, exp3);

    x1 *= x2;    
    ASSERT_EQ(x1, exp4);

    x4 *= x3;
    ASSERT_EQ(x4, exp5);

    x6 *= x5;
    ASSERT_EQ(x6, exp5);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMultiplyEqual_test2) {
    
    NDArray x1('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);    

    const Nd4jLong val1 = 1;
    const float16  val2 = 1.5;
    const double   val3 = 2.2;

    NDArray exp1('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::FLOAT32);
    NDArray exp2('c', {2,2}, {0, 1, 2, 3},  nd4j::DataType::INT32);
    NDArray exp3('c', {2,2}, {0, 1.5, 3, 4.5}, nd4j::DataType::FLOAT32);
    NDArray exp4('c', {2,2}, {0, 1, 3, 4}, nd4j::DataType::INT32);
    NDArray exp5('c', {2,2}, {0, 3.3, 6.6, 9.9}, nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {0, 2, 6, 8}, nd4j::DataType::INT32);
    
    x1 *= val1;
    ASSERT_EQ(x1, exp1);

    x2 *= val1;
    ASSERT_EQ(x2, exp2);

    x1 *= val2;
    ASSERT_EQ(x1, exp3);

    x2 *= val2;    
    ASSERT_EQ(x2, exp4);

    x1 *= val3;
    ASSERT_EQ(x1, exp5);

    x2 *= val3;    
    ASSERT_EQ(x2, exp6);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorDivideEqual_test1) {
    
    NDArray scalar1('c', {0}, {3}, nd4j::DataType::INT32);
    NDArray scalar2('c', {0}, {2.5}, nd4j::DataType::HALF);
    
    NDArray x1('c', {2,3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5},  nd4j::DataType::FLOAT32);
    NDArray x2('c', {3,2}, {10, 20, 30, 40, 50, 60},  nd4j::DataType::INT64);
    NDArray x3('c', {2,2}, {1, 2, 3, 4},  nd4j::DataType::INT64);
    NDArray x4('c', {2},   {0.4, 0.5},  nd4j::DataType::HALF);
    NDArray x5('c', {2,2}, {1, 2, 3, 4},  nd4j::DataType::HALF);
    NDArray x6('c', {2},   {0.4, 0.5},  nd4j::DataType::FLOAT32);

    NDArray exp1('c', {0}, {1},  nd4j::DataType::INT32);
    NDArray exp2('c', {0}, {2.5},  nd4j::DataType::HALF);
    NDArray exp3('c', {3,2}, {6, 8, 8, 8, 9, 9},  nd4j::DataType::INT64);
    NDArray exp4('c', {2,3}, {0.25, 0.3125, 0.4375, 0.5625, 0.611111111, 0.722222222}, nd4j::DataType::FLOAT32);
    NDArray exp5('c', {2,2}, {0.4, 0.25, 0.1333333, 0.125},  nd4j::DataType::HALF);
    
    scalar1 /= scalar2;
    ASSERT_EQ(scalar1, exp1);

    scalar2 /= scalar1;    
    ASSERT_EQ(scalar2, exp2);

    x2 /= x1;
    ASSERT_EQ(x2, exp3);

    x1 /= x2;    
    ASSERT_EQ(x1, exp4);

    x4 /= x3;
    ASSERT_EQ(x4, exp5);

    x6 /= x5;
    ASSERT_EQ(x6, exp5);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorDivideEqual_test2) {
    
    NDArray x1('c', {2,2}, {0, 2, 4, 6}, nd4j::DataType::FLOAT32);
    NDArray x2('c', {2,2}, {0, 2, 4, 6}, nd4j::DataType::INT32);    

    const Nd4jLong val1 = 1;
    const float16  val2 = 2.;
    const double   val3 = 2.2;

    NDArray exp1('c', {2,2}, {0, 2, 4, 6},  nd4j::DataType::FLOAT32);
    NDArray exp2('c', {2,2}, {0, 2, 4, 6},  nd4j::DataType::INT32);
    NDArray exp3('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::FLOAT32);
    NDArray exp4('c', {2,2}, {0, 1, 2, 3}, nd4j::DataType::INT32);
    NDArray exp5('c', {2,2}, {0, 0.45454545, 0.909090909, 1.363636364}, nd4j::DataType::FLOAT32);
    NDArray exp6('c', {2,2}, {0, 0, 0, 1}, nd4j::DataType::INT32);
    
    x1 /= val1;
    ASSERT_EQ(x1, exp1);

    x2 /= val1;
    ASSERT_EQ(x2, exp2);

    x1 /= val2;
    ASSERT_EQ(x1, exp3);

    x2 /= val2;    
    ASSERT_EQ(x2, exp4);

    x1 /= val3;
    ASSERT_EQ(x1, exp5);

    x2 /= val3;    
    ASSERT_EQ(x2, exp6);
}
