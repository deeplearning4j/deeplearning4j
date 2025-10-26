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
// Created by raver119 on 04.08.17.
//
#include <array/NDArray.h>

#include <memory>

#include "ops/specials_sparse.h"
#include "testlayers.h"
using namespace sd;

//////////////////////////////////////////////////////////////////////
class SparseUtilsTest : public NDArrayTests {
 public:
  static const LongType nnz = 40;
  static const int rank = 3;
};

//////////////////////////////////////////////////////////////////////
TEST_F(SparseUtilsTest, SortCOOindices_Test) {
#ifndef SD_CUDA

  sd::LongType* indicesArr = new sd::LongType[nnz * rank]{
      0,  2,  7,  2,  36, 35, 3,  30, 17, 5,  12, 22, 5,  43, 45, 6,  32, 11, 8,  8,  32, 9,  29, 11,
      5,  11, 22, 15, 26, 16, 17, 48, 49, 24, 28, 31, 26, 6,  23, 31, 21, 31, 35, 46, 45, 37, 13, 14,
      6,  38, 18, 7,  28, 20, 8,  29, 39, 8,  32, 30, 9,  42, 43, 11, 15, 18, 13, 18, 45, 29, 26, 39,
      30, 8,  25, 42, 31, 24, 28, 33, 5,  31, 27, 1,  35, 43, 26, 36, 8,  37, 39, 22, 14, 39, 24, 42,
      42, 48, 2,  43, 26, 48, 44, 23, 49, 45, 18, 34, 46, 28, 5,  46, 32, 17, 48, 34, 44, 49, 38, 39,
  };

  sd::LongType* expIndicesArr = new sd::LongType[nnz * rank]{
      0,  2,  7,  2,  36, 35, 3,  30, 17, 5,  11, 22, 5,  12, 22, 5,  43, 45, 6,  32, 11, 6,  38, 18,
      7,  28, 20, 8,  8,  32, 8,  29, 39, 8,  32, 30, 9,  29, 11, 9,  42, 43, 11, 15, 18, 13, 18, 45,
      15, 26, 16, 17, 48, 49, 24, 28, 31, 26, 6,  23, 28, 33, 5,  29, 26, 39, 30, 8,  25, 31, 21, 31,
      31, 27, 1,  35, 43, 26, 35, 46, 45, 36, 8,  37, 37, 13, 14, 39, 22, 14, 39, 24, 42, 42, 31, 24,
      42, 48, 2,  43, 26, 48, 44, 23, 49, 45, 18, 34, 46, 28, 5,  46, 32, 17, 48, 34, 44, 49, 38, 39,
  };

  auto values = NDArrayFactory::create<float>(
      'c', {40}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39});

  auto expValues = NDArrayFactory::create<float>(
      'c', {40}, {0,  1,  2,  8,  3,  4,  5,  16, 17, 6,  18, 19, 7,  20, 21, 22, 9,  10, 11, 12,
                  26, 23, 24, 13, 27, 28, 14, 29, 15, 30, 31, 25, 32, 33, 34, 35, 36, 37, 38, 39});

  sd::sparse::SparseUtils<float>::sortCooIndicesGeneric(indicesArr, reinterpret_cast<float*>(values.buffer()), nnz,
                                                        rank);

  for (int i = 0; i < rank * nnz; ++i) {
    ASSERT_EQ(expIndicesArr[i], indicesArr[i]);
  }

  ASSERT_TRUE(expValues.equalsTo(values));

  delete[] indicesArr;
  delete[] expIndicesArr;

#endif
}

//////////////////////////////////////////////////////////////////////
TEST_F(SparseUtilsTest, RavelIndices_Test) {
#ifndef SD_CUDA

  sd::LongType* indicesArrExp = new sd::LongType[nnz * rank]{
      0,  2,  7,  2,  36, 35, 3,  30, 17, 5,  12, 22, 5,  43, 45, 6,  32, 11, 8,  8,  32, 9,  29, 11,
      5,  11, 22, 15, 26, 16, 17, 48, 49, 24, 28, 31, 26, 6,  23, 31, 21, 31, 35, 46, 45, 37, 13, 14,
      6,  38, 18, 7,  28, 20, 8,  29, 39, 8,  32, 30, 9,  42, 43, 11, 15, 18, 13, 18, 45, 29, 26, 39,
      30, 8,  25, 42, 31, 24, 28, 33, 5,  31, 27, 1,  35, 43, 26, 36, 8,  37, 39, 22, 14, 39, 24, 42,
      42, 48, 2,  43, 26, 48, 44, 23, 49, 45, 18, 34, 46, 28, 5,  46, 32, 17, 48, 34, 44, 49, 38, 39,
  };
  sd::LongType* indicesArr = new sd::LongType[nnz * rank];

  sd::LongType* flatIndicesExp = new sd::LongType[nnz]{
      147,    10955,  14717,  21862,  24055,  27451,  34192,  39841,  21792,  64836,  74809,  102791, 109643, 131701,
      150265, 156324, 27878,  31380,  35669,  35870,  40783,  47268,  55905,  123659, 126585, 178594, 119915, 132091,
      150036, 151797, 165354, 165522, 179762, 182468, 186459, 190294, 195165, 195457, 204024, 208499};

  sd::LongType* flatIndices = new sd::LongType[nnz];

  sd::LongType* shape = new sd::LongType[rank]{50, 60, 70};
  sd::LongType* shapeInfoBuffer = shape::shapeBuffer(rank, sd::DataType::INT64, shape);

  sd::sparse::IndexUtils::ravelMultiIndex(indicesArrExp, flatIndices, nnz, shapeInfoBuffer, ND4J_CLIPMODE_THROW);

  for (int i = 0; i < nnz; ++i) {
    ASSERT_EQ(flatIndicesExp[i], flatIndices[i]);
  }

  sd::sparse::IndexUtils::unravelIndex(indicesArr, flatIndices, nnz, shapeInfoBuffer);

  for (int i = 0; i < nnz * rank; ++i) {
    ASSERT_EQ(indicesArrExp[i], indicesArr[i]);
  }

  shape[2] = 30;
  delete[] shapeInfoBuffer;
  shapeInfoBuffer = shape::shapeBuffer(rank, sd::DataType::INT64, shape);

  try {
    sd::sparse::IndexUtils::ravelMultiIndex(indicesArrExp, flatIndices, nnz, shapeInfoBuffer, ND4J_CLIPMODE_THROW);
    FAIL();
  } catch (const std::runtime_error& e) {
    // pass
  }

  delete[] indicesArrExp;
  delete[] indicesArr;
  delete[] flatIndicesExp;
  delete[] flatIndices;
  delete[] shape;
  delete[] shapeInfoBuffer;

#endif
}
