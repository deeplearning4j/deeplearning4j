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
// Created by raver119 on 13.01.2018.
//
#include <array/ArrayOptions.hXX>
#include <array/NDArray.h>

#include "testlayers.h"

using namespace sd;

class ArrayOptionsTests : public NDArrayTests {
 public:
  LongType shape[8] = {2, 5, 5, 5, 1, 0, 1, 99};
};

TEST_F(ArrayOptionsTests, TestShape_Basic_0) {
  shape[5] = 1;

  ASSERT_TRUE(ArrayOptions::isNewFormat(shape));
  ASSERT_FALSE(ArrayOptions::isSparseArray(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_1) {
  shape[5] = 2;

  ASSERT_TRUE(ArrayOptions::isNewFormat(shape));
  ASSERT_TRUE(ArrayOptions::isSparseArray(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_2) {
  shape[5] = 258;

  ASSERT_TRUE(ArrayOptions::isNewFormat(shape));

  ASSERT_TRUE(ArrayOptions::isSparseArray(shape));
  ASSERT_EQ(SpaceType::CONTINUOUS, ArrayOptions::spaceType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_3) {
  ASSERT_EQ(0, shape::extra(shape));

  ASSERT_EQ(SpaceType::CONTINUOUS, ArrayOptions::spaceType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_4) {
  ArrayOptions::setPropertyBits(shape, {ARRAY_HALF, ARRAY_QUANTIZED});

  auto dtype = ArrayOptions::dataType(shape);

  ASSERT_FALSE(ArrayOptions::isSparseArray(shape));
  ASSERT_TRUE(sd::DataType::HALF == ArrayOptions::dataType(shape));
  ASSERT_EQ(sd::ArrayType::DENSE, ArrayOptions::arrayType(shape));
  ASSERT_EQ(sd::SpaceType::QUANTIZED, ArrayOptions::spaceType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_5) {
  ArrayOptions::setPropertyBits(shape, {ARRAY_SPARSE, ARRAY_INT, ARRAY_CSC});

  ASSERT_TRUE(ArrayOptions::isSparseArray(shape));
  ASSERT_TRUE(sd::DataType::INT32 == ArrayOptions::dataType(shape));
  ASSERT_EQ(sd::SparseType::CSC, ArrayOptions::sparseType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_6) {
  ArrayOptions::setPropertyBits(shape, {ARRAY_EMPTY, ARRAY_INT, ARRAY_CSC});

  ASSERT_EQ(sd::ArrayType::EMPTY, ArrayOptions::arrayType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_7) {
  ArrayOptions::setDataType(shape, FLOAT32);
  ArrayOptions::setDataType(shape, FLOAT32);

  ASSERT_EQ(sd::DataType::FLOAT32, ArrayOptions::dataType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_8) {
  ArrayOptions::setDataType(shape, DOUBLE);
  ArrayOptions::setDataType(shape, FLOAT32);

  ASSERT_EQ(sd::DataType::FLOAT32, ArrayOptions::dataType(shape));
}

TEST_F(ArrayOptionsTests, TestShape_Basic_9) {
  ArrayOptions::setDataType(shape, FLOAT32);
  ArrayOptions::setDataType(shape, DOUBLE);

  ASSERT_EQ(sd::DataType::DOUBLE, ArrayOptions::dataType(shape));
}
