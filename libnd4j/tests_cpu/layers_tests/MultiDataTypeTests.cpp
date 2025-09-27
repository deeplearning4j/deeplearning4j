/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <array/ArrayOptions.hXX>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/MmulHelper.h>
#include <ops/declarable/headers/broadcastable.h>

#include "testlayers.h"

using namespace sd;

class MultiDataTypeTests : public NDArrayTests {
 public:
};

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_1) {
  auto dtype = DataTypeUtils::pickPairwiseResultType(INT32, FLOAT32);

  ASSERT_EQ(sd::FLOAT32, dtype);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_2) {
  auto dtype = DataTypeUtils::pickPairwiseResultType(INT32, DOUBLE);
  ASSERT_EQ(sd::DOUBLE, dtype);

  ASSERT_EQ(sd::DOUBLE, DataTypeUtils::pickPairwiseResultType(sd::DOUBLE, sd::INT32));
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, DataTypeUtils_Test_3) {
  auto dtype = DataTypeUtils::pickPairwiseResultType(FLOAT32, DOUBLE);
  ASSERT_EQ(sd::FLOAT32, dtype);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  auto y = NDArrayFactory::create<double>('c', {2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
  auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

  auto z = x + y;

  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  auto y = NDArrayFactory::create<double>(2.0);
  auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

  auto z = x * y;

  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_3) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
  auto y = NDArrayFactory::create<double>(2.0);
  auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

  auto z = x * y;

  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_4) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  auto y = NDArrayFactory::create<float>(2.0);
  auto e = NDArrayFactory::create<double>('c', {2, 3}, {0.0f, 2.0f, 4.0f, 6.0f, 8.0f, 10.0f});

  auto z = x * y;

  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_5) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
  auto y = NDArrayFactory::create<LongType>(2);
  auto e = NDArrayFactory::create<int>('c', {2, 3}, {0, 2, 4, 6, 8, 10});

  auto z = x * y;

  ASSERT_EQ(e, z);
}

TEST_F(MultiDataTypeTests, Basic_Test_7) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<int>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
  auto y = NDArrayFactory::create<float>('c', {2, 3}, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
  auto e = NDArrayFactory::create<float>('c', {2, 3}, {0.f, 2.f, 4.f, 6.f, 8.f, 10.f});

  ops::add op;
  auto result = op.evaluate({&x, &y});
  ASSERT_EQ(sd::Status::OK, result.status());

  auto z = result.at(0);

  ASSERT_EQ(e, *z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, Basic_Test_6) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  auto x = NDArrayFactory::create<LongType>('c', {2, 3}, {0, 1, 2, 3, 4, 5});
  auto y = NDArrayFactory::create<int>(2);
  auto e = NDArrayFactory::create<LongType>('c', {2, 3}, {0, 2, 4, 6, 8, 10});

  auto z = x * y;

  ASSERT_EQ(e, z);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test1) {
  NDArray x('c', {2, 3}, {0, 1, 2, 3, 4, 5}, UINT8);
  NDArray exp('c', {2, 3}, {10, 10, 10, 10, 10, 10}, UINT8);

  const double number = 10.8;
  x = number;

  ASSERT_EQ(x, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test2) {
  NDArray x('c', {2, 3}, {0, 1, 2, 3, 4, 5}, INT64);
  NDArray exp('c', {2, 3}, {1, 1, 1, 1, 1, 1}, INT64);

  const bool number = 1000;
  x = number;

  ASSERT_EQ(x, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_number_test3) {
  NDArray x('c', {2, 3}, {0, 1, 0, 1, 0, 1}, BOOL);
  NDArray exp('c', {2, 3}, {1, 1, 1, 1, 1, 1}, BOOL);

  const int number = 1000;
  x = number;

  ASSERT_EQ(x, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_repeat_test1) {
  NDArray x('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, HALF);
  NDArray y('c', {2, 4}, HALF);
  NDArray exp('c', {2, 4}, {0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5}, HALF);

  x.repeat(1, {2}, y);

  ASSERT_EQ(y, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_bufferAsT_test1) {
  NDArray x('f', {2}, {1.5, 3.5}, FLOAT32);
  NDArray y('c', {}, std::vector<double>{1.5}, FLOAT32);

  const int* buffX = x.bufferAsT<int>();
  const int* buffY = y.bufferAsT<int>();

  ASSERT_EQ(*buffX, *buffY);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_assign_test1) {
  NDArray x('c', {2, 2}, {0, 1, 2, 3}, UINT8);
  NDArray exp('c', {2, 2}, {10, 10, 20, 20}, UINT8);

  NDArray scalar1('c', {}, std::vector<double>{10.5}, FLOAT32);
  NDArray scalar2('c', {}, std::vector<double>{20.8}, DOUBLE);

  x(0, {0}).assign(scalar1);
  x(1, {0}).assign(scalar2);

  ASSERT_EQ(x, exp);

  x.assign(exp);

  ASSERT_EQ(x, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test1) {
  NDArray x('f', {2, 2}, {0, 1.5, 2.5, 3.5}, HALF);
  NDArray exp1('c', {}, std::vector<double>{3}, INT64);
  NDArray exp2('c', {1, 1}, std::vector<double>{1}, INT64);
  NDArray exp3('c', {2}, std::vector<double>{1, 2}, INT64);

  std::vector<LongType> empty;
  std::vector<LongType>  dimOne = {1};
  auto scalar1 = x.reduceAlongDimension(reduce::CountNonZero,&empty /*whole range*/);
  ASSERT_EQ(scalar1, exp1);

  auto scalar2 = x.reduceAlongDimension(reduce::CountZero, &empty /*whole range*/, true);
  ASSERT_EQ(scalar2, exp2);

  auto scalar3 = x.reduceAlongDimension(reduce::CountNonZero,&dimOne);
  ASSERT_EQ(scalar3, exp3);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test2) {
  NDArray x('c', {2, 2}, {0, 1, 2, 3}, INT32);
  NDArray exp1('c', {}, std::vector<double>{1.5}, FLOAT32);
  NDArray exp2('c', {2}, {0.5, 2.5}, FLOAT32);
  std::vector<LongType> empty;
  std::vector<LongType>  dimOne = {1};
  auto scalar1 = x.reduceAlongDimension(reduce::Mean, &empty /*whole range*/);
  ASSERT_EQ(scalar1, exp1);

  auto scalar2 = x.reduceAlongDimension(reduce::Mean, &dimOne);
  ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test3) {
  NDArray x('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, HALF);
  NDArray exp1('c', {}, std::vector<double>{8.}, HALF);
  NDArray exp2('c', {2}, {2., 6.}, HALF);
  std::vector<LongType> empty;
  std::vector<LongType>  dimOne = {1};
  auto scalar1 = x.reduceAlongDimension(reduce::Sum, &empty /*whole range*/);
  ASSERT_EQ(scalar1, exp1);

  auto scalar2 = x.reduceAlongDimension(reduce::Sum, &dimOne);
  ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceAlongDimension_test4) {
  NDArray x('c', {2, 2}, {10.5, 1.5, -2.5, -3.5}, HALF);
  NDArray exp1('c', {}, std::vector<double>{1}, BOOL);
  NDArray exp2('c', {2}, std::vector<double>{1, 0}, BOOL);
  std::vector<LongType> empty;
  std::vector<LongType>  dimOne = {1};
  auto scalar1 = x.reduceAlongDimension(reduce::IsPositive, &empty /*whole range*/);
  ASSERT_EQ(scalar1, exp1);

  auto scalar2 = x.reduceAlongDimension(reduce::IsPositive, &dimOne);
  ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_varianceNumber_test1) {
  NDArray x('f', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray exp1('c', {}, std::vector<double>{1.666666667}, FLOAT32);
  NDArray exp2('c', {}, std::vector<double>{1.118033989}, FLOAT32);

  auto scalar1 = x.varianceNumber(variance::SummaryStatsVariance);
  ASSERT_EQ(scalar1, exp1);

  auto scalar2 = x.varianceNumber(variance::SummaryStatsStandardDeviation, false);
  ASSERT_EQ(scalar2, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlus_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, FLOAT32);
  NDArray x3('c', {2}, {-1, -2}, FLOAT32);

  NDArray exp('c', {2, 2}, {-1, -1, 1, 1}, FLOAT32);

  ASSERT_EQ(x1 + x2, exp);
  ASSERT_EQ(x1 + x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlus_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, HALF);
  const double val1 = -2;
  const int val2 = -2;
  NDArray exp1('c', {2, 2}, {-2, -1, 0, 1}, DOUBLE);
  NDArray exp2('c', {2, 2}, {-2, -1, 0, 1}, FLOAT32);
  NDArray exp3('c', {2, 2}, {-2, -1, 0, 1}, HALF);

  ASSERT_EQ(x1 + val1, exp1);
  ASSERT_EQ(val1 + x1, exp1);

  ASSERT_EQ(x2 + val2, exp2);
  ASSERT_EQ(val2 + x2, exp2);

  ASSERT_EQ(x3 + val1, exp3);
  ASSERT_EQ(val1 + x3, exp3);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinus_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, HALF);
  NDArray x3('c', {2}, {-1, -2}, HALF);

  NDArray exp('c', {2, 2}, {1, 3, 3, 5}, HALF);

  ASSERT_EQ(x1 - x2, exp);
  ASSERT_EQ(x1 - x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMinus_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, HALF);
  const double val1 = 2;
  const int val2 = 2;
  NDArray exp1('c', {2, 2}, {-2, -1, 0, 1}, DOUBLE);
  NDArray exp2('c', {2, 2}, {2, 1, 0, -1}, DOUBLE);
  NDArray exp3('c', {2, 2}, {-2, -1, 0, 1}, FLOAT32);
  NDArray exp4('c', {2, 2}, {-2, -1, 0, 1}, HALF);
  NDArray exp5('c', {2, 2}, {2, 1, 0, -1}, FLOAT32);
  NDArray exp6('c', {2, 2}, {2, 1, 0, -1}, HALF);

  ASSERT_EQ(x1 - val1, exp1);
  ASSERT_EQ(val1 - x1, exp2);

  ASSERT_EQ(x2 - val2, exp3);
  ASSERT_EQ(val2 - x2, exp5);

  ASSERT_EQ(x3 - val1, exp4);
  ASSERT_EQ(val1 - x3, exp6);
}

//////////////////////////////////////////////////////////////////////////////// multiply
TEST_F(MultiDataTypeTests, ndarray_operatorMultiply_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {-1, -2, -1, -2}, DOUBLE);
  NDArray x3('c', {2}, {-1, -2}, DOUBLE);

  NDArray exp('c', {2, 2}, {0, -2, -2, -6}, DOUBLE);

  ASSERT_EQ(x1 * x2, exp);
  ASSERT_EQ(x1 * x3, exp);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorMultiply_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, HALF);
  const double val1 = -2;
  const int val2 = -2;
  NDArray exp1('c', {2, 2}, {0, -2, -4, -6}, DOUBLE);
  NDArray exp2('c', {2, 2}, {0, -2, -4, -6}, FLOAT32);
  NDArray exp3('c', {2, 2}, {0, -2, -4, -6}, HALF);

  ASSERT_EQ(x1 * val1, exp1);
  ASSERT_EQ(val1 * x1, exp1);

  ASSERT_EQ(x2 * val2, exp2);
  ASSERT_EQ(val2 * x2, exp2);

  ASSERT_EQ(x3 * val1, exp3);
  ASSERT_EQ(val1 * x3, exp3);
}

//////////////////////////////////////////////////////////////////////////////// multiply
TEST_F(MultiDataTypeTests, ndarray_operatorDivide_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {4, 1, 2, 3}, HALF);
  NDArray x2('c', {2, 2}, {-1, -2, -1, -9}, DOUBLE);
  NDArray x3('c', {2}, {-1, -2}, FLOAT32);

  NDArray exp1('c', {2, 2}, {-4, -0.5, -2, -0.3333333}, HALF);
  NDArray exp2('c', {2, 2}, {-0.25, -2, -0.5, -0.666667}, HALF);

  ASSERT_EQ(x1 / x2, exp1);
  ASSERT_EQ(x3 / x1, exp2);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorDivide_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {1, 2, 3, 4}, INT64);
  NDArray x2('c', {2, 2}, {1, 2, 3, 4}, FLOAT32);
  NDArray x3('c', {2, 2}, {1, 2, 3, 4}, HALF);
  const double val1 = 2;
  const int val2 = -2;
  NDArray exp1('c', {2, 2}, {0.5, 1, 1.5, 2}, DOUBLE);
  NDArray exp2('c', {2, 2}, {2, 1, 0.666667, 0.5}, DOUBLE);
  NDArray exp3('c', {2, 2}, {0, -1, -1, -2}, INT64);
  NDArray exp4('c', {2, 2}, {-2, -1, 0., 0.}, INT64);
  NDArray exp5('c', {2, 2}, {-0.5, -1, -1.5, -2}, FLOAT32);
  NDArray exp6('c', {2, 2}, {-2, -1, -0.666667, -0.5}, FLOAT32);
  NDArray exp7('c', {2, 2}, {0.5, 1, 1.5, 2}, HALF);
  NDArray exp8('c', {2, 2}, {2, 1, 0.666667, 0.5}, HALF);

  ASSERT_EQ(x1 / val1, exp1);
  ASSERT_EQ(val1 / x1, exp2);

  ASSERT_EQ(x1 / val2, exp3);
  ASSERT_EQ(val2 / x1, exp4);

  ASSERT_EQ(x2 / val2, exp5);
  ASSERT_EQ(val2 / x2, exp6);

  ASSERT_EQ(x3 / val1, exp7);
  ASSERT_EQ(val1 / x3, exp8);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_operatorPlusEqual_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray scalar1('c', {0}, std::vector<double>{4}, INT32);
  NDArray scalar2('c', {0}, std::vector<double>{1.5}, HALF);

  NDArray x1('c', {2, 3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, {10, 20, 30, 40, 50, 60}, INT64);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x4('c', {2}, {0.4, 0.5}, HALF);
  NDArray x5('c', {2, 2}, {0, 1, 2, 3}, HALF);
  NDArray x6('c', {2}, {0.4, 0.5}, FLOAT32);

  NDArray exp1('c', {0}, std::vector<double>{5}, INT32);
  NDArray exp2('c', {0}, std::vector<double>{6.5}, HALF);
  NDArray exp3('c', {3, 2}, {11, 22, 33, 44, 55, 66}, INT64);
  NDArray exp4('c', {2, 3}, {12.5, 24.5, 36.5, 48.5, 60.5, 72.5}, FLOAT32);
  NDArray exp5('c', {2, 2}, {0.4, 1.5, 2.4, 3.5}, HALF);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, INT32);

  const LongType val1 = 1;
  const float16 val2 = 1.5;
  const double val3 = 2.2;

  NDArray exp1('c', {2, 2}, {1, 2, 3, 4}, FLOAT32);
  NDArray exp2('c', {2, 2}, {1, 2, 3, 4}, INT32);
  NDArray exp3('c', {2, 2}, {2.5, 3.5, 4.5, 5.5}, FLOAT32);
  NDArray exp4('c', {2, 2}, {2, 3, 4.5, 5}, INT32);
  NDArray exp5('c', {2, 2}, {4.7, 5.7, 6.7, 7.7}, FLOAT32);
  NDArray exp6('c', {2, 2}, {4, 5, 6, 7}, INT32);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray scalar1('c', {0}, std::vector<double>{4}, INT32);
  NDArray scalar2('c', {0}, std::vector<double>{1.5}, HALF);

  NDArray x1('c', {2, 3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, {10, 20, 30, 40, 50, 60}, INT64);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x4('c', {2}, {0.4, 0.5}, HALF);
  NDArray x5('c', {2, 2}, {0, 1, 2, 3}, HALF);
  NDArray x6('c', {2}, {0.4, 0.5}, FLOAT32);

  NDArray exp1('c', {0}, std::vector<double>{2}, INT32);
  NDArray exp2('c', {0}, std::vector<double>{-0.5}, HALF);
  NDArray exp3('c', {3, 2}, {8, 17, 26, 35, 44, 53}, INT64);
  NDArray exp4('c', {2, 3}, {-6.5, -14.5, -22.5, -30.5, -38.5, -46.5}, FLOAT32);
  NDArray exp5('c', {2, 2}, {0.4, -0.5, -1.6, -2.5}, HALF);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, INT32);

  const LongType val1 = 1;
  const float16 val2 = 1.5;
  const double val3 = 2.2;

  NDArray exp1('c', {2, 2}, {-1, 0, 1, 2}, FLOAT32);
  NDArray exp2('c', {2, 2}, {-1, 0, 1, 2}, INT32);
  NDArray exp3('c', {2, 2}, {-2.5, -1.5, -0.5, 0.5}, FLOAT32);
  NDArray exp4('c', {2, 2}, {-2., -1., 0., 0.}, INT32);
  NDArray exp5('c', {2, 2}, {-4.7, -3.7, -2.7, -1.7}, FLOAT32);
  NDArray exp6('c', {2, 2}, {-4, -3, -2, -2}, INT32);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray scalar1('c', {0}, std::vector<double>{3}, INT32);
  NDArray scalar2('c', {0}, std::vector<double>{2.5}, HALF);

  NDArray x1('c', {2, 3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, {1, 2, 3, 4, 5, 6}, INT64);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x4('c', {2}, {0.4, 0.5}, HALF);
  NDArray x5('c', {2, 2}, {0, 1, 2, 3}, HALF);
  NDArray x6('c', {2}, {0.4, 0.5}, FLOAT32);

  NDArray exp1('c', {0}, std::vector<double>{7}, INT32);
  NDArray exp2('c', {0}, std::vector<double>{17.5}, HALF);
  NDArray exp3('c', {3, 2}, {1, 5, 10, 18, 27, 39}, INT64);
  NDArray exp4('c', {2, 3}, {1.5, 12.5, 35, 81, 148.5, 253.5}, FLOAT32);
  NDArray exp5('c', {2, 2}, {0., 0.5, 0.8, 1.5}, HALF);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, INT32);

  const LongType val1 = 1;
  const float16 val2 = 1.5;
  const double val3 = 2.2;

  NDArray exp1('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray exp2('c', {2, 2}, {0, 1, 2, 3}, INT32);
  NDArray exp3('c', {2, 2}, {0, 1.5, 3, 4.5}, FLOAT32);
  NDArray exp4('c', {2, 2}, {0, 1, 3, 4}, INT32);
  NDArray exp5('c', {2, 2}, {0, 3.3, 6.6, 9.9}, FLOAT32);
  NDArray exp6('c', {2, 2}, {0, 2, 6, 8}, INT32);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray scalar1('c', {0}, std::vector<double>{3}, INT32);
  NDArray scalar2('c', {0}, std::vector<double>{2.5}, HALF);

  NDArray x1('c', {2, 3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, {10, 20, 30, 40, 50, 60}, INT64);
  NDArray x3('c', {2, 2}, {1, 2, 3, 4}, INT64);
  NDArray x4('c', {2}, {0.4, 0.5}, HALF);
  NDArray x5('c', {2, 2}, {1, 2, 3, 4}, HALF);
  NDArray x6('c', {2}, {0.4, 0.5}, FLOAT32);

  NDArray exp1('c', {0}, std::vector<double>{1}, INT32);
  NDArray exp2('c', {0}, std::vector<double>{2.5}, HALF);
  NDArray exp3('c', {3, 2}, {6, 8, 8, 8, 9, 9}, INT64);
  NDArray exp4('c', {2, 3}, {0.25, 0.3125, 0.4375, 0.5625, 0.611111111, 0.722222222}, FLOAT32);
  NDArray exp5('c', {2, 2}, {0.4, 0.25, 0.1333333, 0.125}, HALF);

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
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 2, 4, 6}, FLOAT32);
  NDArray x2('c', {2, 2}, {0, 2, 4, 6}, INT32);

  const LongType val1 = 1;
  const float16 val2 = 2.;
  const double val3 = 2.2;

  NDArray exp1('c', {2, 2}, {0, 2, 4, 6}, FLOAT32);
  NDArray exp2('c', {2, 2}, {0, 2, 4, 6}, INT32);
  NDArray exp3('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray exp4('c', {2, 2}, {0, 1, 2, 3}, INT32);
  NDArray exp5('c', {2, 2}, {0, 0.45454545, 0.909090909, 1.363636364}, FLOAT32);
  NDArray exp6('c', {2, 2}, {0, 0, 0, 1}, INT32);

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

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceNumberFloat_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);

  NDArray exp1('c', {0}, std::vector<double>{1.5}, FLOAT32);
  NDArray exp2('c', {0}, std::vector<double>{2}, HALF);
  NDArray exp3('c', {0}, std::vector<double>{2}, DOUBLE);
  NDArray exp4('c', {0}, std::vector<double>{0.25}, FLOAT32);

  NDArray scalar = x1.reduceNumber(reduce::Mean);
  ASSERT_EQ(scalar, exp1);
  x1.reduceNumber(reduce::Mean, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x2.reduceNumber(reduce::Mean);
  ASSERT_EQ(scalar, exp2);
  x2.reduceNumber(reduce::Mean, scalar);
  ASSERT_EQ(scalar, exp2);

  scalar = x3.reduceNumber(reduce::Mean);
  ASSERT_EQ(scalar, exp3);
  x3.reduceNumber(reduce::Mean, scalar);
  ASSERT_EQ(scalar, exp3);

  scalar = x4.reduceNumber(reduce::Mean);
  ASSERT_EQ(scalar, exp4);
  x4.reduceNumber(reduce::Mean, scalar);
  ASSERT_EQ(scalar, exp4);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceNumberSame_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);

  NDArray exp1('c', {0}, std::vector<double>{6}, INT64);
  NDArray exp2('c', {0}, std::vector<double>{8}, HALF);
  NDArray exp3('c', {0}, std::vector<double>{8}, DOUBLE);
  NDArray exp4('c', {0}, std::vector<double>{1}, BOOL);

  NDArray scalar = x1.reduceNumber(reduce::Sum);
  ASSERT_EQ(scalar, exp1);
  x1.reduceNumber(reduce::Sum, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x2.reduceNumber(reduce::Sum);
  ASSERT_EQ(scalar, exp2);
  x2.reduceNumber(reduce::Sum, scalar);
  ASSERT_EQ(scalar, exp2);

  scalar = x3.reduceNumber(reduce::Sum);
  ASSERT_EQ(scalar, exp3);
  x3.reduceNumber(reduce::Sum, scalar);
  ASSERT_EQ(scalar, exp3);

  scalar = x4.reduceNumber(reduce::Sum);
  ASSERT_EQ(scalar, exp4);
  x4.reduceNumber(reduce::Sum, scalar);
  ASSERT_EQ(scalar, exp4);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceNumberBool_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, -1, 2, -3}, INT64);
  NDArray x2('c', {2, 2}, {0.5, -1.5, 2.5, -3.5}, HALF);
  NDArray x3('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {-2, -1, 0, 1}, BOOL);

  NDArray exp1('c', {0}, std::vector<double>{1}, BOOL);

  NDArray scalar = x1.reduceNumber(reduce::IsFinite);
  ASSERT_EQ(scalar, exp1);
  x1.reduceNumber(reduce::IsFinite, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x2.reduceNumber(reduce::IsFinite);
  ASSERT_EQ(scalar, exp1);
  x2.reduceNumber(reduce::IsFinite, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x3.reduceNumber(reduce::IsFinite);
  ASSERT_EQ(scalar, exp1);
  x3.reduceNumber(reduce::IsFinite, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x4.reduceNumber(reduce::IsFinite);
  ASSERT_EQ(scalar, exp1);
  x4.reduceNumber(reduce::IsFinite, scalar);
  ASSERT_EQ(scalar, exp1);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_reduceNumberLong_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0.5, 1.5, 2.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0.5, -1.5, 0, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);

  NDArray exp1('c', {0}, std::vector<double>{3}, INT64);
  NDArray exp2('c', {0}, std::vector<double>{4}, INT64);
  NDArray exp3('c', {0}, std::vector<double>{3}, INT64);
  NDArray exp4('c', {0}, std::vector<double>{2}, INT64);

  NDArray scalar = x1.reduceNumber(reduce::CountNonZero);
  ASSERT_EQ(scalar, exp1);
  x1.reduceNumber(reduce::CountNonZero, scalar);
  ASSERT_EQ(scalar, exp1);

  scalar = x2.reduceNumber(reduce::CountNonZero);
  ASSERT_EQ(scalar, exp2);
  x2.reduceNumber(reduce::CountNonZero, scalar);
  ASSERT_EQ(scalar, exp2);

  scalar = x3.reduceNumber(reduce::CountNonZero);
  ASSERT_EQ(scalar, exp3);
  x3.reduceNumber(reduce::CountNonZero, scalar);
  ASSERT_EQ(scalar, exp3);

  scalar = x4.reduceNumber(reduce::CountNonZero);
  ASSERT_EQ(scalar, exp4);
  x4.reduceNumber(reduce::CountNonZero, scalar);
  ASSERT_EQ(scalar, exp4);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_indexReduceNumber_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT32);
  NDArray x2('c', {2, 2}, {0.5, 1.5, -4.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0, -1, 0, 1}, BOOL);

  NDArray exp1('c', {0}, std::vector<double>{3}, INT64);
  NDArray exp2('c', {0}, std::vector<double>{2}, INT64);
  NDArray exp3('c', {0}, std::vector<double>{1}, INT64);

  NDArray scalar = x1.indexReduceNumber(indexreduce::IndexAbsoluteMax);
  ASSERT_EQ(scalar, exp1);

  scalar = x2.indexReduceNumber(indexreduce::IndexAbsoluteMax);
  ASSERT_EQ(scalar, exp2);

  scalar = x3.indexReduceNumber(indexreduce::IndexAbsoluteMax);
  ASSERT_EQ(scalar, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTransformFloat_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 4, 9, 16}, INT64);
  NDArray x2('c', {2, 2}, {0, 2.25, 6.25, 12.25}, HALF);
  NDArray x3('c', {2, 2}, {0, 2.25, 6.25, 12.25}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);

  NDArray exp1('c', {2, 2}, {0, 2, 3, 4}, FLOAT32);
  NDArray exp2('c', {2, 2}, {0, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray exp3('c', {2, 2}, {0, 1.5, 2.5, 3.5}, HALF);
  NDArray exp4('c', {2, 2}, {0, 1, 0, 1}, HALF);

  NDArray result1('c', {2, 2}, FLOAT32);
  NDArray result2('c', {2, 2}, DOUBLE);
  NDArray result3('c', {2, 2}, HALF);

  x1.applyTransform(transform::Sqrt, result1);
  ASSERT_EQ(result1, exp1);

  x2.applyTransform(transform::Sqrt, result2);
  ASSERT_EQ(result2, exp2);

  x3.applyTransform(transform::Sqrt, result3);
  ASSERT_EQ(result3, exp3);

  x4.applyTransform(transform::Sqrt, result3);
  ASSERT_EQ(result3, exp4);

  x2.applyTransform(transform::Sqrt, x2);
  ASSERT_EQ(x2, exp3);

  x3.applyTransform(transform::Sqrt, x3);
  ASSERT_EQ(x3, exp2);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTransformSame_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1.5, 2.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);
  NDArray x5('c', {2, 3}, {0, 1.5, 2.5, 3.5, 4.5, 5.5}, DOUBLE);

  NDArray exp1('c', {2, 2}, {0, 1, 4, 9}, INT64);
  NDArray exp2('c', {2, 2}, {0, 2.25, 6.25, 12.25}, HALF);
  NDArray exp3('c', {2, 2}, {0, 2.25, 6.25, 12.25}, DOUBLE);
  NDArray exp4('c', {2, 2}, {0, 1, 0, 1}, BOOL);
  NDArray exp5('c', {3, 2}, {0, 2.25, 6.25, 12.25, 20.25, 30.25}, DOUBLE);

  NDArray result1('c', {2, 2}, INT64);
  NDArray result2('c', {2, 2}, HALF);
  NDArray result3('c', {2, 2}, DOUBLE);
  NDArray result4('c', {2, 2}, BOOL);
  NDArray result5('c', {3, 2}, DOUBLE);

  x1.applyTransform(transform::Square, result1);
  ASSERT_EQ(result1, exp1);

  x2.applyTransform(transform::Square, result2);
  ASSERT_EQ(result2, exp2);

  x3.applyTransform(transform::Square, result3);
  ASSERT_EQ(result3, exp3);

  x4.applyTransform(transform::Square, result4);
  ASSERT_EQ(result4, exp4);

  x2.applyTransform(transform::Square, x2);
  ASSERT_EQ(x2, exp2);

  x3.applyTransform(transform::Square, x3);
  ASSERT_EQ(x3, exp3);

  x5.applyTransform(transform::Square, result5);
  ASSERT_EQ(result5, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTransformBool_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1.5, 2.5, 3.5}, HALF);
  NDArray x3('c', {2, 2}, {0, 1.5, 2.5, 3.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);
  NDArray x5('c', {2, 3}, {0, 1.5, 2.5, 3.5, 4.5, 5.5}, DOUBLE);

  NDArray exp1('c', {2, 2}, {0, 0, 0, 1}, BOOL);
  NDArray exp2('c', {2, 2}, {0, 1, 0, 0}, BOOL);
  NDArray exp3('c', {3, 2}, {0, 0, 0, 0, 0, 1}, BOOL);

  NDArray result1('c', {2, 2}, BOOL);
  NDArray result2('c', {3, 2}, BOOL);

}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTransformStrict_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, HALF);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, FLOAT32);
  NDArray x3('c', {2, 2}, {0, 1, 2, 3}, DOUBLE);
  NDArray x4('c', {2, 3}, {0, 1, 2, 3, 4, 5}, DOUBLE);

  NDArray exp1('c', {2, 2}, {0, 3, 12, 27}, HALF);
  NDArray exp2('c', {2, 2}, {0, 3, 12, 27}, FLOAT32);
  NDArray exp3('c', {2, 2}, {0, 3, 12, 27}, DOUBLE);
  NDArray exp4('c', {3, 2}, {0, 3, 12, 27, 48, 75}, DOUBLE);
  NDArray exp5('c', {2, 3}, {0, 3, 12, 27, 48, 75}, DOUBLE);

  NDArray result1('c', {2, 2}, HALF);
  NDArray result2('c', {2, 2}, FLOAT32);
  NDArray result3('c', {2, 2}, DOUBLE);
  NDArray result4('c', {3, 2}, DOUBLE);

  x1.applyTransform(transform::CubeDerivative, result1);
  ASSERT_EQ(result1, exp1);

  x2.applyTransform(transform::CubeDerivative, result2);
  ASSERT_EQ(result2, exp2);

  x3.applyTransform(transform::CubeDerivative, result3);
  ASSERT_EQ(result3, exp3);

  x4.applyTransform(transform::CubeDerivative, result4);
  ASSERT_EQ(result4, exp4);

  x1.applyTransform(transform::CubeDerivative, x1);
  ASSERT_EQ(x1, exp1);

  x2.applyTransform(transform::CubeDerivative, x2);
  ASSERT_EQ(x2, exp2);

  x3.applyTransform(transform::CubeDerivative, x3);
  ASSERT_EQ(x3, exp3);

  x4.applyTransform(transform::CubeDerivative, x4);
  ASSERT_EQ(x4, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyPairwiseTransform_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 3}, {0, 1, 2, 3, 4, 5}, INT32);
  NDArray x2('c', {2, 3}, {0, 1, 2, 3, 4, 5}, FLOAT32);
  NDArray x3('c', {2, 3}, {0, 1, 0, 1, 0, 0}, BOOL);
  NDArray x4('c', {3, 2}, {0.5, 1.5, 2.5, 3.5, 4.5, 0}, DOUBLE);
  NDArray x5('c', {3, 2}, INT32);
  NDArray x6('c', {2, 3}, DOUBLE);

  NDArray exp1('c', {2, 3}, {0, 2, 4, 6, 8, 5}, INT32);
  NDArray exp2('c', {2, 3}, {0.5, 2.5, 4.5, 6.5, 8.5, 5.}, FLOAT32);
  NDArray exp3('c', {2, 3}, {1, 1, 1, 1, 1, 0}, BOOL);
  NDArray exp4('c', {2, 3}, {0.5, 2.5, 4.5, 6.5, 8.5, 5.}, DOUBLE);
  NDArray exp5('c', {3, 2}, {0, 2, 4, 6, 8, 5}, INT32);

  x1.applyPairwiseTransform(pairwise::Add, x4, x5);
  ASSERT_EQ(x5, exp5);

  x1.applyPairwiseTransform(pairwise::Add, x4, x6);
  ASSERT_EQ(x6, exp4);

  x1.applyPairwiseTransform(pairwise::Add, x4);
  ASSERT_EQ(x1, exp1);

  x2.applyPairwiseTransform(pairwise::Add, x4);
  ASSERT_EQ(x2, exp2);

  x3.applyPairwiseTransform(pairwise::Add, x4);
  ASSERT_EQ(x3, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyPairwiseTransform_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 3}, {1, 1, 2, 3, 4, 5}, INT32);
  NDArray x2('c', {3, 2}, {1, 0, 2, 0, 4, 0}, INT32);
  NDArray x3('c', {3, 2}, {0.5, 1.5, 2.5, 3, 4.5, 0}, DOUBLE);
  NDArray x4('c', {2, 3}, {0.5, 1., 2.5, 3, 4., 0}, DOUBLE);
  NDArray x5('c', {3, 2}, {0, 1, 0, 1, 0, 1}, BOOL);
  NDArray x6('c', {2, 3}, {1, 1, 1, 0, 1, 0}, BOOL);

  NDArray x7('c', {3, 2}, BOOL);
  NDArray x8('c', {2, 3}, BOOL);

  NDArray exp1('c', {3, 2}, {1, 0, 1, 0, 1, 0}, BOOL);
  NDArray exp2('c', {2, 3}, {1, 0, 1, 1, 0, 1}, BOOL);
  NDArray exp3('c', {2, 3}, {0, 1, 0, 0, 0, 0}, BOOL);

  x1.applyPairwiseTransform(pairwise::EqualTo, x2, x7);
  ASSERT_EQ(x7, exp1);

  x3.applyPairwiseTransform(pairwise::EqualTo, x4, x8);
  ASSERT_EQ(x8, exp2);

  x5.applyPairwiseTransform(pairwise::EqualTo, x6, x8);
  ASSERT_EQ(x8, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyBroadcast_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 3}, {10, 20, 30, 40, 50, 60}, INT32);
  NDArray x2('c', {2}, {1, 2}, INT64);
  NDArray x3('c', {2, 3}, INT32);
  NDArray x4('c', {2}, {1, 2}, FLOAT32);
  NDArray x5('c', {2, 3}, FLOAT32);
  NDArray x6('c', {2}, {1, 1}, BOOL);

  NDArray exp1('c', {2, 3}, {11, 21, 31, 42, 52, 62}, INT32);
  NDArray exp2('c', {2, 3}, {11, 21, 31, 42, 52, 62}, FLOAT32);
  NDArray exp3('c', {2, 3}, {11, 21, 31, 41, 51, 61}, INT32);
  std::vector<LongType> dimZero = {0};
  x1.applyBroadcast(broadcast::Add, &dimZero, x2, x3);
  ASSERT_EQ(x3, exp1);

  x1.applyBroadcast(broadcast::Add, &dimZero, x4, x5);
  ASSERT_EQ(x5, exp2);

  x1.applyBroadcast(broadcast::Add, &dimZero, x6, x3);
  ASSERT_EQ(x3, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyBroadcast_test2) {
  NDArray x1('c', {2, 3}, {10, 20, 30, 40, 50, 60}, INT32);
  NDArray x2('c', {2}, {10, 60}, INT32);
  NDArray x3('c', {2, 3}, BOOL);

  NDArray x4('c', {2, 3}, {0, 0, 0, 0, 0, 1}, BOOL);
  NDArray x5('c', {2}, {0, 1}, BOOL);

  NDArray exp1('c', {2, 3}, {1, 0, 0, 0, 0, 1}, BOOL);
  NDArray exp2('c', {2, 3}, {1, 1, 1, 0, 0, 1}, BOOL);

  std::vector<LongType> zero = {0};

  x1.applyBroadcast(broadcast::EqualTo, &zero, x2, x3);
  ASSERT_EQ(x3, exp1);

  x4.applyBroadcast(broadcast::EqualTo, &zero, x5, x3);
  ASSERT_EQ(x3, exp2);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTrueBroadcast_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {10, 20, 30, 40}, INT32);
  NDArray x2('c', {2}, {1, 2}, HALF);
  NDArray x3('c', {2, 2}, HALF);

  NDArray x4('c', {2}, {1, 2}, INT64);
  NDArray x5('c', {2, 2}, INT32);

  NDArray x6('c', {2, 2}, {0, 1, 0, 1}, BOOL);
  NDArray x7('c', {2}, {1, 2}, INT64);
  NDArray x8('c', {2, 2}, BOOL);

  NDArray x13('c', {0}, std::vector<double>{3}, INT64);
  NDArray x14('c', {0}, std::vector<double>{1.5}, DOUBLE);
  NDArray x15(DOUBLE);
  NDArray x16('c', {2, 2}, DOUBLE);

  NDArray exp1('c', {2, 2}, {11, 22, 31, 42}, HALF);
  NDArray exp2('c', {2, 2}, {11, 22, 31, 42}, INT32);
  NDArray exp3('c', {2, 2}, {1, 1, 1, 1}, BOOL);
  NDArray exp4('c', {0}, std::vector<double>{4.5}, DOUBLE);
  NDArray exp5('c', {2, 2}, {11.5, 21.5, 31.5, 41.5}, DOUBLE);

  x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x2, x3);
  ASSERT_EQ(x3, exp1);

  x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x4, x5);
  ASSERT_EQ(x5, exp2);

  x6.applyTrueBroadcast(BroadcastOpsTuple::Add(), x7, x8);
  ASSERT_EQ(x8, exp3);

  auto x9 = x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x2);
  ASSERT_EQ(x9, exp1);

  auto x10 = x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x4);
  ASSERT_EQ(x10, exp2);

  auto x11 = x6.applyTrueBroadcast(BroadcastOpsTuple::Add(), x7);
  ASSERT_EQ(x11, exp3);

  auto x12 = x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x2);
  ASSERT_EQ(x12, exp1);

  x13.applyTrueBroadcast(BroadcastOpsTuple::Add(), x14, x15);
  ASSERT_EQ(x15, exp4);

  x1.applyTrueBroadcast(BroadcastOpsTuple::Add(), x14, x16);
  ASSERT_EQ(x16, exp5);

  x14.applyTrueBroadcast(BroadcastOpsTuple::Add(), x1, x16);
  ASSERT_EQ(x16, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTrueBroadcast_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {10, 20, 30, 40}, HALF);
  NDArray x2('c', {2}, {10, 40}, HALF);
  NDArray x3('c', {2, 2}, BOOL);
  NDArray x4('c', {0}, std::vector<double>{10}, HALF);
  NDArray x5('c', {0}, std::vector<double>{20}, HALF);
  NDArray x6(BOOL);

  NDArray exp1('c', {2, 2}, {1, 0, 0, 1}, BOOL);
  NDArray exp2('c', {2, 2}, {1, 0, 0, 0}, BOOL);
  NDArray exp3('c', {0}, std::vector<double>{0}, BOOL);

  x1.applyTrueBroadcast(BroadcastBoolOpsTuple(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), x2,
                        x3);
  ASSERT_EQ(x3, exp1);

  x1.applyTrueBroadcast(BroadcastBoolOpsTuple(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), x4,
                        x3);
  ASSERT_EQ(x3, exp2);

  x4.applyTrueBroadcast(BroadcastBoolOpsTuple(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), x1,
                        x3);
  ASSERT_EQ(x3, exp2);

  x5.applyTrueBroadcast(BroadcastBoolOpsTuple(scalar::EqualTo, pairwise::EqualTo, broadcast::EqualTo), x4,
                        x6);
  ASSERT_EQ(x6, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyScalar_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1.5, 2.5, 3.5}, FLOAT32);
  NDArray x3('c', {2, 2}, DOUBLE);
  NDArray x4('c', {2, 2}, {0, 1, 0, 1}, BOOL);

  NDArray exp1('c', {2, 2}, {1, 2, 3, 4}, INT64);
  NDArray exp2('c', {2, 2}, {1.5, 2.5, 3.5, 4.5}, DOUBLE);
  NDArray exp3('c', {2, 2}, {0.1, 1.6, 2.6, 3.6}, FLOAT32);
  NDArray exp4('c', {2, 2}, {1.1, 2.1, 1.1, 2.1}, DOUBLE);
  NDArray exp5('c', {2, 2}, {1, 1, 1, 1}, BOOL);

  x1.applyScalar<int>(scalar::Add, 1, x1);
  ASSERT_EQ(x1, exp1);

  x1.applyScalar<double>(scalar::Add, 0.5, x3);
  ASSERT_EQ(x3, exp2);

  x2.applyScalar<double>(scalar::Add, 0.1, x2);
  ASSERT_EQ(x2, exp3);

  x4.applyScalar<double>(scalar::Add, 1.1, x3);
  ASSERT_EQ(x3, exp4);

  x4.applyScalar<LongType>(scalar::Add, 1, x4);
  ASSERT_EQ(x4, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyScalar_test2) {
  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, INT64);
  NDArray x2('c', {2, 2}, {0, 1.5, 2.5, 3.5}, FLOAT32);
  NDArray x3('c', {2, 2}, {0, 1, 1, 0}, BOOL);
  NDArray x4('c', {2, 2}, BOOL);

  NDArray exp1('c', {2, 2}, {0, 1, 0, 0}, BOOL);
  NDArray exp2('c', {2, 2}, {0, 1, 1, 0}, BOOL);

  x1.applyScalar<LongType>(scalar::EqualTo, 1, x4);
  ASSERT_EQ(x4, exp1);

  x2.applyScalar<float>(scalar::EqualTo, 1.5, x4);
  ASSERT_EQ(x4, exp1);

  x3.applyScalar<bool>(scalar::EqualTo, true, x4);
  ASSERT_EQ(x4, exp2);
}

#ifndef SD_CUDA
//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyLambda_test1) {
  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, sd::DataType::DOUBLE);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, sd::DataType::INT64);
  NDArray x3('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x4('c', {2, 2}, sd::DataType::DOUBLE);
  NDArray x5('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x6('c', {2, 2}, {0, -1, -1, 0.1}, sd::DataType::BOOL);
  NDArray x7('c', {2, 2}, sd::DataType::BOOL);

  const float item1 = 0.1;
  const double item2 = 0.1;
  auto func1 = [=](float elem) { return elem + item1; };
  auto func2 = [=](int elem) { return elem + item1; };
  auto func3 = [=](int elem) { return elem + item2; };
  auto func4 = [=](double elem) { return elem + item1; };
  auto func5 = [=](float elem) { return elem - (int)1; };

  NDArray exp1('c', {2, 2}, {0.1, 1.1, 2.1, 3.1}, sd::DataType::DOUBLE);
  NDArray exp2('c', {2, 2}, {0, 1, 2, 3}, sd::DataType::INT64);
  NDArray exp3('c', {2, 2}, {0.1, 1.1, 2.1, 3.1}, sd::DataType::FLOAT32);
  NDArray exp4('c', {2, 2}, {0.1, 1.6, 2.6, 3.6}, sd::DataType::FLOAT32);
  NDArray exp5('c', {2, 2}, {1, 0, 0, 0}, sd::DataType::BOOL);

  x1.applyLambda<double>(func1, x4);
  ASSERT_EQ(x4, exp1);

  x2.applyLambda<sd::LongType>(func1, x2);
  ASSERT_EQ(x2, exp2);

  x2.applyLambda<sd::LongType>(func2, x2);
  ASSERT_EQ(x2, exp2);

  x3.applyLambda<float>(func3, x3);
  ASSERT_EQ(x3, exp3);

  x5.applyLambda<float>(func4, x5);
  ASSERT_EQ(x5, exp4);

  x6.applyLambda<bool>(func5, x7);
  ASSERT_EQ(x7, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyIndexedLambda_test1) {
  NDArray x1('c', {2, 2}, {0, 1, 2, 3}, sd::DataType::DOUBLE);
  NDArray x2('c', {2, 2}, {0, 1, 2, 3}, sd::DataType::INT64);
  NDArray x3('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x4('c', {2, 2}, sd::DataType::DOUBLE);
  NDArray x5('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x6('c', {2, 2}, {1, -1, -1, 0.1}, sd::DataType::BOOL);
  NDArray x7('c', {2, 2}, sd::DataType::BOOL);

  const float item1 = 0.1;
  const double item2 = 0.1;
  auto func1 = [=](sd::LongType idx, float elem) { return idx + elem + item1; };
  auto func2 = [=](sd::LongType idx, int elem) { return idx + elem + item1; };
  auto func3 = [=](sd::LongType idx, int elem) { return idx + elem + item2; };
  auto func4 = [=](sd::LongType idx, double elem) { return idx + elem + item1; };
  auto func5 = [=](sd::LongType idx, float elem) { return idx + elem - (int)1; };

  NDArray exp1('c', {2, 2}, {0.1, 2.1, 4.1, 6.1}, sd::DataType::DOUBLE);
  NDArray exp2('c', {2, 2}, {0, 2, 4, 6}, sd::DataType::INT64);
  NDArray exp3('c', {2, 2}, {0.1, 2.1, 4.1, 6.1}, sd::DataType::FLOAT32);
  NDArray exp4('c', {2, 2}, {0.1, 2.6, 4.6, 6.6}, sd::DataType::FLOAT32);
  NDArray exp5('c', {2, 2}, {0, 1, 1, 1}, sd::DataType::BOOL);
  NDArray exp6('c', {2, 2}, {0, 3, 6, 9}, sd::DataType::INT64);

  x1.applyIndexedLambda<double>(func1, x4);
  ASSERT_EQ(x4, exp1);

  x2.applyIndexedLambda<sd::LongType>(func1, x2);
  ASSERT_EQ(x2, exp2);

  x2.applyIndexedLambda<sd::LongType>(func2, x2);
  ASSERT_EQ(x2, exp6);

  x3.applyIndexedLambda<float>(func3, x3);
  ASSERT_EQ(x3, exp3);

  x5.applyIndexedLambda<float>(func4, x5);
  ASSERT_EQ(x5, exp4);

  x6.applyIndexedLambda<bool>(func5, x7);
  ASSERT_EQ(x7, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyPairwiseLambda_test1) {
  NDArray x1('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::DOUBLE);
  NDArray x2('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::INT64);
  NDArray x3('c', {2, 2}, {0., 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x4('c', {2, 2}, sd::DataType::DOUBLE);
  NDArray x5('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x6('c', {2, 2}, {0.1, -1, -1, 0.1}, sd::DataType::BOOL);
  NDArray x7('c', {2, 2}, sd::DataType::BOOL);
  NDArray other1('c', {2, 2}, {0.1, 0.1, 0.1, 0.1}, sd::DataType::FLOAT32);
  NDArray other2('c', {2, 2}, {0.1, 0.1, 0.1, 0.1}, sd::DataType::DOUBLE);
  NDArray other3('c', {2, 2}, {0., -1, -2, -3}, sd::DataType::INT64);
  NDArray other4('c', {2, 2}, {1, 0, 0.1, 0}, sd::DataType::BOOL);

  auto func1 = [](float elem1, float elem2) { return elem1 + elem2; };
  auto func2 = [](int elem1, float elem2) { return elem1 + elem2; };
  auto func3 = [](int elem1, double elem2) { return elem1 + elem2; };
  auto func4 = [](double elem1, float elem2) { return elem1 + elem2; };
  auto func5 = [](float elem1, int elem2) { return elem1 - elem2; };

  NDArray exp1('c', {2, 2}, {0.1, 1.1, 2.1, 3.1}, sd::DataType::DOUBLE);
  NDArray exp2('c', {2, 2}, {0., 0, 0, 0}, sd::DataType::INT64);
  NDArray exp3('c', {2, 2}, {0.1, 1.1, 2.1, 3.1}, sd::DataType::FLOAT32);
  NDArray exp4('c', {2, 2}, {0.1, 1.6, 2.6, 3.6}, sd::DataType::FLOAT32);
  NDArray exp5('c', {2, 2}, {0., 1, 0, 1}, sd::DataType::BOOL);

  x1.applyPairwiseLambda<double>(other2, func1, x4);
  ASSERT_EQ(x4, exp1);

  x2.applyPairwiseLambda<sd::LongType>(other3, func1, x2);
  ASSERT_EQ(x2, exp2);

  x2.applyPairwiseLambda<sd::LongType>(other3, func2, x2);
  ASSERT_EQ(x2, other3);

  x3.applyPairwiseLambda<float>(other1, func3, x3);
  ASSERT_EQ(x3, exp3);

  x5.applyPairwiseLambda<float>(other1, func4, x5);
  ASSERT_EQ(x5, exp4);

  x6.applyPairwiseLambda<bool>(other4, func5, x7);
  ASSERT_EQ(x7, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyIndexedPairwiseLambda_test1) {
  NDArray x1('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::DOUBLE);
  NDArray x2('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::INT64);
  NDArray x3('c', {2, 2}, {0., 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x4('c', {2, 2}, sd::DataType::DOUBLE);
  NDArray x5('c', {2, 2}, {0, 1.5, 2.5, 3.5}, sd::DataType::FLOAT32);
  NDArray x6('c', {2, 2}, {0.1, -1, -1, 0.1}, sd::DataType::BOOL);
  NDArray x7('c', {2, 2}, sd::DataType::BOOL);
  NDArray other1('c', {2, 2}, {0.1, 0.1, 0.1, 0.1}, sd::DataType::FLOAT32);
  NDArray other2('c', {2, 2}, {0.1, 0.1, 0.1, 0.1}, sd::DataType::DOUBLE);
  NDArray other3('c', {2, 2}, {0., -1, -2, -3}, sd::DataType::INT64);
  NDArray other4('c', {2, 2}, {1, 0, 0.1, 0}, sd::DataType::BOOL);

  auto func1 = [](sd::LongType idx, float elem1, float elem2) { return elem1 + elem2 + idx; };
  auto func2 = [](sd::LongType idx, int elem1, float elem2) { return elem1 + elem2 + idx; };
  auto func3 = [](sd::LongType idx, int elem1, double elem2) { return elem1 + elem2 + idx; };
  auto func4 = [](sd::LongType idx, double elem1, float elem2) { return elem1 + elem2 + idx; };
  auto func5 = [](sd::LongType idx, float elem1, int elem2) { return elem1 - elem2 + idx; };

  NDArray exp1('c', {2, 2}, {0.1, 2.1, 4.1, 6.1}, sd::DataType::DOUBLE);
  NDArray exp2('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::INT64);
  NDArray exp3('c', {2, 2}, {0.1, 2.1, 4.1, 6.1}, sd::DataType::FLOAT32);
  NDArray exp4('c', {2, 2}, {0.1, 2.6, 4.6, 6.6}, sd::DataType::FLOAT32);
  NDArray exp5('c', {2, 2}, {0., 1, 1, 1}, sd::DataType::BOOL);

  x1.applyIndexedPairwiseLambda<double>(other2, func1, x4);
  ASSERT_EQ(x4, exp1);

  x2.applyIndexedPairwiseLambda<sd::LongType>(other3, func1, x2);
  ASSERT_EQ(x2, exp2);

  x2.applyIndexedPairwiseLambda<sd::LongType>(other3, func2, x2);
  ASSERT_EQ(x2, exp2);

  x3.applyIndexedPairwiseLambda<float>(other1, func3, x3);
  ASSERT_EQ(x3, exp3);

  x5.applyIndexedPairwiseLambda<float>(other1, func4, x5);
  ASSERT_EQ(x5, exp4);

  x6.applyIndexedPairwiseLambda<bool>(other4, func5, x7);
  ASSERT_EQ(x7, exp5);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyTriplewiseLambda_test1) {
  NDArray x1('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::DOUBLE);
  NDArray x2('c', {2, 2}, {0., -1, -2, -3}, sd::DataType::DOUBLE);
  NDArray x3('c', {2, 2}, {0, -1.5, -2.5, -3.5}, sd::DataType::DOUBLE);
  NDArray x4('c', {2, 2}, sd::DataType::DOUBLE);

  NDArray x5('c', {2, 2}, {0., 1, 2, 3}, sd::DataType::INT32);
  NDArray x6('c', {2, 2}, {0., -1, -2, -3}, sd::DataType::INT32);
  NDArray x7('c', {2, 2}, {0., 10, 20, 30}, sd::DataType::INT32);

  NDArray x8('c', {2, 2}, {0., 1, 0, 1}, sd::DataType::BOOL);
  NDArray x9('c', {2, 2}, {1., 1, 0, 1}, sd::DataType::BOOL);
  NDArray x10('c', {2, 2}, {0., 0, 0, 0}, sd::DataType::BOOL);

  auto func1 = [](double elem1, float elem2, int elem3) { return elem1 + elem2 + elem3; };
  auto func2 = [](float elem1, float elem2, float elem3) { return elem1 + elem2 + elem3; };
  auto func3 = [](int elem1, int elem2, int elem3) { return elem1 + elem2 + elem3; };
  auto func4 = [](bool elem1, bool elem2, bool elem3) { return elem1 + elem2 + elem3; };

  NDArray exp('c', {2, 2}, {1., 1, 0, 1}, sd::DataType::BOOL);

  x1.applyTriplewiseLambda<double>(x2, x3, func1, x4);
  ASSERT_EQ(x4, x2);

  x1.applyTriplewiseLambda<double>(x2, x3, func2, x1);
  ASSERT_EQ(x1, x3);

  x5.applyTriplewiseLambda<int>(x6, x7, func3, x5);
  ASSERT_EQ(x5, x7);

  x8.applyTriplewiseLambda<bool>(x9, x10, func4, x8);
  ASSERT_EQ(x8, exp);
}

#endif

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyIndexReduce_test1) {
  NDArray x1('c', {2, 3}, {0, 1, 2, 3, 4, 5}, DOUBLE);
  NDArray exp1('c', {}, std::vector<double>{5}, INT64);
  NDArray exp2('c', {2}, {2, 2}, INT64);
  NDArray exp3('c', {3}, {1, 1, 1}, INT64);

  std::vector<LongType> zerOone = {0, 1};
  std::vector<LongType> zero = {0};
  std::vector<LongType> one = {1};

  NDArray scalar = x1.applyIndexReduce(indexreduce::IndexMax, &zerOone);
  ASSERT_EQ(scalar, exp1);

  NDArray vec1 = x1.applyIndexReduce(indexreduce::IndexMax, &one);
  ASSERT_EQ(vec1, exp2);

  NDArray vec2 = x1.applyIndexReduce(indexreduce::IndexMax, &zero);
  ASSERT_EQ(vec2, exp3);
}

//////////////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, ndarray_applyIndexReduce_test2) {
  NDArray x1('c', {2, 3}, {0, 1, 2, 3, 4, 5}, DOUBLE);
  NDArray scalar('c', {}, std::vector<double>{5}, INT64);
  NDArray vec1('c', {2}, {2, 2}, INT64);
  NDArray vec2('c', {3}, {1, 1, 1}, INT64);
  NDArray exp1('c', {}, std::vector<double>{5}, INT64);
  NDArray exp2('c', {2}, {2, 2}, INT64);
  NDArray exp3('c', {3}, {1, 1, 1}, INT64);

  std::vector<LongType> empty;
  std::vector<LongType>  dimOne = {1};

  std::vector<LongType> zeroOne = {0, 1};
  x1.applyIndexReduce(indexreduce::IndexMax, scalar, &zeroOne);
  ASSERT_EQ(scalar, exp1);

  x1.applyIndexReduce(indexreduce::IndexMax, vec1, &dimOne);
  ASSERT_EQ(vec1, exp2);

  std::vector<LongType> zero = {0};

  x1.applyIndexReduce(indexreduce::IndexMax, vec2, &zero);
  ASSERT_EQ(vec2, exp3);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, applyReduce3_test1) {
  NDArray x1('c', {2, 2}, {1, 2, 3, 4}, INT32);
  NDArray x2('c', {2, 2}, {-1, -2, -3, -4}, INT32);
  NDArray x3('c', {2, 2}, {1.5, 1.5, 1.5, 1.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {1, 2, 3, 4}, DOUBLE);
  NDArray exp1('c', {}, std::vector<double>{-30}, FLOAT32);
  NDArray exp2('c', {}, std::vector<double>{15}, DOUBLE);

  auto result = x1.applyReduce3(reduce3::Dot, x2);
  ASSERT_EQ(result, exp1);

  result = x3.applyReduce3(reduce3::Dot, x4);
  ASSERT_EQ(result, exp2);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, applyReduce3_test2) {
  NDArray x1('c', {2, 2}, {1, 2, 3, 4}, INT32);
  NDArray x2('c', {2, 2}, {-1, -2, -3, -4}, INT32);
  NDArray x3('c', {2, 2}, {1.5, 1.5, 1.5, 1.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {1, 2, 3, 4}, DOUBLE);
  NDArray x5('c', {2, 3}, {1, 2, 3, 4, 5, 6}, INT32);
  NDArray x6('c', {2, 3}, {-6, -5, -4, -3, -2, -1}, INT32);
  NDArray x7('c', {2, 3}, {1.5, 1.5, 1.5, 1.5, 1.5, 1.5}, DOUBLE);
  NDArray x8('c', {2, 3}, {1, 2, 3, 4, 5, 6}, DOUBLE);

  NDArray exp1('c', {}, std::vector<double>{-30}, FLOAT32);
  NDArray exp2('c', {}, std::vector<double>{15}, DOUBLE);
  NDArray exp3('c', {3}, {-18, -20, -18}, FLOAT32);
  NDArray exp4('c', {2}, {-28, -28}, FLOAT32);
  NDArray exp5('c', {3}, {7.5, 10.5, 13.5}, DOUBLE);
  NDArray exp6('c', {2}, {9, 22.5}, DOUBLE);

  auto result = x1.applyReduce3(reduce3::Dot, x2, {0, 1});
  ASSERT_EQ(result, exp1);

  result = x3.applyReduce3(reduce3::Dot, x4, {0, 1});
  ASSERT_EQ(result, exp2);

  result = x5.applyReduce3(reduce3::Dot, x6, std::vector<LongType>({0}));
  ASSERT_EQ(result, exp3);

  result = x5.applyReduce3(reduce3::Dot, x6, std::vector<LongType>({1}));
  ASSERT_EQ(result, exp4);

  result = x8.applyReduce3(reduce3::Dot, x7, std::vector<LongType>({0}));
  ASSERT_EQ(result, exp5);

  result = x8.applyReduce3(reduce3::Dot, x7, std::vector<LongType>({1}));
  ASSERT_EQ(result, exp6);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, applyAllReduce3_test1) {
  NDArray x1('c', {2, 2}, {1, 2, 3, 4}, INT32);
  NDArray x2('c', {2, 3}, {-1, 1, -1, 1, -1, 1}, INT32);
  NDArray x3('c', {2, 3}, {1.5, 1.5, 1.5, 1.5, 1.5, 1.5}, DOUBLE);
  NDArray x4('c', {2, 2}, {1, 2, 3, 4}, DOUBLE);
  NDArray exp1('c', {2, 3}, {2, -2, 2, 2, -2, 2}, FLOAT32);
  NDArray exp2('c', {2, 3}, {6, 6, 6, 9, 9, 9}, DOUBLE);

  std::vector<LongType> zero = {0};

  auto result = x1.applyAllReduce3(reduce3::Dot, x2,&zero);
  ASSERT_EQ(result, exp1);

  result = x4.applyAllReduce3(reduce3::Dot, x3, &zero);
  ASSERT_EQ(result, exp2);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, RowCol_test1) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 3}, {1, 2, 3, 4, 5, 6}, INT32);
  NDArray x2('c', {2}, {0.5, 0.6}, FLOAT32);
  NDArray x3('c', {3}, {1.5, 1.6, 1.7}, FLOAT32);
  NDArray x4('c', {2, 3}, {1, 2, 3, 4, 5, 6}, DOUBLE);
  NDArray x5('c', {2, 3}, {1, 2, 3, 4, 5, 6}, INT32);

  NDArray exp1('c', {2, 3}, {2, 3, 4, 5, 6, 7}, INT32);
  NDArray exp2('c', {2, 3}, {0, 1, 2, 3, 4, 5}, INT32);
  NDArray exp3('c', {2, 3}, {1.5, 2.5, 3.5, 4.6, 5.6, 6.6}, DOUBLE);
  NDArray exp4('c', {2, 3}, {0, 1, 1, 2, 3, 3}, INT32);

  x1.addiRowVector(x3);
  ASSERT_EQ(x1, exp1);

  x1.addiColumnVector(x2);
  ASSERT_EQ(x1, exp1);

  x4.addiColumnVector(x2);
  ASSERT_EQ(x4, exp3);

  x5.muliColumnVector(x2);
  ASSERT_EQ(x5, exp4);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, RowCol_test2) {
  if (!Environment::getInstance().isExperimentalBuild()) return;

  NDArray x1('c', {2, 3}, {1, 2, 3, 4, 5, 6}, INT32);
  NDArray x2('c', {2}, {0.5, 0.6}, FLOAT32);
  NDArray x3('c', {3}, {1.5, 1.6, 1.7}, FLOAT32);
  NDArray x4('c', {2, 3}, FLOAT32);
  NDArray x5('c', {3}, {1, 2, 3}, INT64);
  NDArray x6('c', {2, 3}, INT32);
  NDArray x7('c', {3}, {1.5, 1.6, 1.7}, DOUBLE);
  NDArray x8('c', {2, 3}, {1, 2, 3, 4, 5, 6}, FLOAT32);
  NDArray x9('c', {3}, {1, 2, 3}, DOUBLE);
  NDArray x10('c', {2, 3}, DOUBLE);

  NDArray exp1('c', {2, 3}, {2.5, 3.6, 4.7, 5.5, 6.6, 7.7}, FLOAT32);
  NDArray exp2('c', {2, 3}, {2, 4, 6, 5, 7, 9}, INT32);
  NDArray exp3('c', {2, 3}, {-0.5, 0.4, 1.3, 2.5, 3.4, 4.3}, FLOAT32);
  NDArray exp4('c', {2, 3}, {1, 4, 9, 4, 10, 18}, DOUBLE);
  NDArray exp5('c', {2, 3}, {1, 1, 1, 4, 2.5, 2}, DOUBLE);
  NDArray exp6('c', {2, 3}, {1.5, 2.5, 3.5, 4.6, 5.6, 6.6}, FLOAT32);

  x1.addRowVector(x3, x4);
  ASSERT_EQ(x4, exp1);

  x1.addRowVector(x5, x6);
  ASSERT_EQ(x6, exp2);

  x8.subRowVector(x7, x4);
  ASSERT_EQ(x4, exp3);

  x1.mulRowVector(x9, x10);
  ASSERT_EQ(x10, exp4);

  x1.divRowVector(x9, x10);
  ASSERT_EQ(x10, exp5);

  x1.addColumnVector(x2, x4);
  ASSERT_EQ(x4, exp6);
}

//////////////////////////////////////////////////////////////////////


#if defined(HAS_INT) && defined(HAS_DOUBLE) && defined(HAS_FLOAT32)
//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, asT_test1) {
  NDArray x1('c', {2}, {1.5, 2.5}, sd::DataType::FLOAT32);

  NDArray exp1('c', {2}, {1, 2}, sd::DataType::INT32);
  NDArray exp2('c', {2}, {1.5, 2.5}, sd::DataType::DOUBLE);

  auto result = new NDArray(x1.asT<int>());
  ASSERT_EQ(*result, exp1);
  delete result;

  result = new NDArray(x1.asT<double>());
  ASSERT_EQ(*result, exp2);
  delete result;

  result = new NDArray(x1.asT(sd::DataType::INT32));
  ASSERT_EQ(*result, exp1);
  delete result;

  result = new NDArray(x1.asT(sd::DataType::DOUBLE));
  ASSERT_EQ(*result, exp2);
  delete result;
}
#endif
//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, assign_test2) {
  NDArray x1('c', {2, 3}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, INT32);
  NDArray x3('c', {3, 2}, DOUBLE);
  NDArray x4('c', {3, 2}, BOOL);
  NDArray x5('c', {2, 3}, {1.5, 2.5, 0, 4.5, 5.5, 6.5}, FLOAT32);

  NDArray exp1('c', {3, 2}, {1, 2, 3, 4, 5, 6}, INT32);
  NDArray exp2('c', {3, 2}, {1.5, 2.5, 3.5, 4.5, 5.5, 6.5}, DOUBLE);
  NDArray exp3('c', {3, 2}, {1, 1, 0, 1, 1, 1}, BOOL);

  x2.assign(x1);
  ASSERT_EQ(x2, exp1);

  x3.assign(x1);
  ASSERT_EQ(x3, exp2);

  x4.assign(x5);
  ASSERT_EQ(x4, exp3);
}

TEST_F(MultiDataTypeTests, Test_Cast_1) {
  auto first = NDArrayFactory::create<float>('c', {10});
  auto asBool = NDArrayFactory::create<bool>('c', {10});
  auto _not = NDArrayFactory::create<bool>('c', {10});
  auto asFloat = NDArrayFactory::create<float>('c', {10});
  auto exp = NDArrayFactory::create<float>('c', {10});
  exp.assign(0.0f);

  asBool.assign(first);

  asBool.applyScalar(scalar::Not, false, _not);

  asFloat.assign(_not);
  ASSERT_EQ(exp, asFloat);
}

TEST_F(MultiDataTypeTests, Test_Cast_2) {
  auto first = NDArrayFactory::create<float>('c', {10});
  auto asBool = NDArrayFactory::create<bool>('c', {10});
  auto _not = NDArrayFactory::create<bool>('c', {10});
  auto asFloat = NDArrayFactory::create<float>('c', {10});
  auto exp = NDArrayFactory::create<float>('c', {10});
  exp.assign(1.0f);

  asBool.assign(first);

  asBool.applyTransform(transform::Not, _not);


  asFloat.assign(_not);

  ASSERT_EQ(exp, asFloat);
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, divide_bool_test1) {
  NDArray x1('c', {2, 3}, {1.5, 0, 3.5, 0, 5.5, 6.5}, FLOAT32);
  NDArray x2('c', {3, 2}, {1, 1, 0, 1, 0, 1}, BOOL);
  NDArray x3('c', {2, 3}, FLOAT32);
  NDArray x4('c', {2}, BOOL);

  try {
    NDArray x3 = x1 / x2;
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }

  try {
    x1 /= x2;
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }

  try {
    NDArray x3 = 150. / x2;
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }

  try {
    x1.divRowVector(x4, x3);
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }

  try {

    std::vector<LongType> one = {1};
    x1.applyBroadcast(broadcast::FloorDiv,&one, x4, x3);
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }

  try {
    x1.applyTrueBroadcast(BROADCAST(FloorMod), x2, x3);
  } catch (std::exception& message) {
    ASSERT_TRUE(1);
  }
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, aaa) {
  NDArray z('c', {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, DOUBLE);
  z.permutei({1, 0});

  RandomGenerator gen(119, 5);
  ExtraArguments extras({1.5, 2.5});

  NativeOpExecutioner::execRandom(LaunchContext::defaultContext(), random::UniformDistribution, &gen, z.buffer(),
                                  z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(),
                                  extras.argumentsAsT<double>());
}

//////////////////////////////////////////////////////////////////////
TEST_F(MultiDataTypeTests, assign_2) {
  NDArray x('c', {4}, {1.5, 2.5, 3.5, 4.5}, FLOAT32);
  NDArray y('c', {4}, INT32);
  NDArray expected('c', {4}, {1, 2, 3, 4}, INT32);

  y.assign(x);

  ASSERT_TRUE(expected.equalsTo(&y));
}
