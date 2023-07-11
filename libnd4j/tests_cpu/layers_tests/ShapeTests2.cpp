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
// Created by agibsonccc on 1/6/17.
//
#include <helpers/ShapeBuilders.h>
#include <helpers/TAD.h>
#include <helpers/data_gen.h>

#include "testinclude.h"

class OnesTest : public NDArrayTests {
 public:
  sd::LongType shapeBuffer[12] = {4, 4, 3, 1, 1, 3, 1, 1, 1, 0, 1, 99};
  sd::LongType dimension[3] = {0, 2, 3};
  sd::LongType tadAssertionShape[10] = {3, 1, 1, 4, 1, 1, 3, 0, 3, 99};
  int dimensionLength = 3;
};

class LabelTest : public NDArrayTests {
 public:
  float labels[450] = {
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
      1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  sd::LongType shapeInfo[8] = {2, 150, 3, 1, 150, 16384, 1, 102};
  sd::LongType dimension[1] = {1};
  int dimensionLength = 1;
  sd::LongType tadShapeInfoAssert[8] = {2, 1, 3, 1, 150, 16384, 150, 102};
};
class ThreeDTest : public NDArrayTests {
 public:
  sd::LongType shape[3] = {3, 4, 5};
  sd::LongType *shapeBuffer;
  ThreeDTest() { shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 3, shape); }
  ~ThreeDTest() { delete[] shapeBuffer; }
};

class VectorTest : public NDArrayTests {};

class NumTadTests : public NDArrayTests {
 public:
  sd::LongType shape[3] = {3, 4, 5};
  sd::LongType dimension = 0;
};

class ShapeTest : public NDArrayTests {
 public:
  sd::LongType vectorShape[2] = {1, 2};
};

class MatrixTest : public NDArrayTests {
 public:
  int rows = 3;
  int cols = 4;
  int rank = 2;
  sd::LongType dims[2] = {0, 1};
  sd::LongType expectedShapes[2][2] = {{1, 3}, {1, 4}};
  sd::LongType expectedStrides[2][2] = {{1, 4}, {1, 1}};
};

class TADStall : public NDArrayTests {
 public:
  sd::LongType shape[4] = {3, 3, 4, 5};
  sd::LongType dimensions[3] = {1, 2, 3};
};

class TensorOneDimTest : public NDArrayTests {
 public:
  int rows = 3;
  int cols = 4;
  sd::LongType dim2 = 5;
  int rank = 3;
  sd::LongType dims[3] = {0, 1, 2};
  sd::LongType expectedShapes[3][2] = {{1, 3}, {1, 4}, {1, 5}};
  sd::LongType expectedStrides[3][2] = {{1, 20}, {1, 5}, {1, 1}};
};

class TensorTwoDimTest : public NDArrayTests {
 public:
  // From a 3d array:
  int rows = 3;
  int cols = 4;
  int dim2 = 5;
  int dimensionLength = 2;
  sd::LongType dims[3][2] = {{0, 1}, {0, 2}, {1, 2}};

  sd::LongType shape[3]{rows, cols, dim2};

  // Along dimension 0,1: expect matrix with shape [rows,cols]
  // Along dimension 0,2: expect matrix with shape [rows,dim2]
  // Along dimension 1,2: expect matrix with shape [cols,dim2]
  sd::LongType expectedShapes[3][2] = {{rows, cols}, {rows, dim2}, {cols, dim2}};

  sd::LongType expectedStrides[3][2] = {{20, 5}, {20, 1}, {5, 1}};
};

class TensorTwoFromFourDDimTest : public NDArrayTests {
 public:
  // From a 3d array:
  int rows = 3;
  int cols = 4;
  int dim2 = 5;
  int dim3 = 6;
  sd::LongType shape[4] = {rows, cols, dim2, dim3};
  int dimensionLength = 2;
  // Along dimension 0,1: expect matrix with shape [rows,cols]
  // Along dimension 0,2: expect matrix with shape [rows,dim2]
  // Along dimension 0,3: expect matrix with shape [rows,dim3]
  // Along dimension 1,2: expect matrix with shape [cols,dim2]
  // Along dimension 1,3: expect matrix with shape [cols,dim3]
  // Along dimension 2,3: expect matrix with shape [dim2,dim3]

  sd::LongType dims[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

  sd::LongType expectedShapes[6][2] = {{rows, cols}, {rows, dim2}, {rows, dim3},
                                       {cols, dim2}, {cols, dim3}, {dim2, dim3}};

  sd::LongType expectedStrides[6][2] = {{120, 30}, {120, 6}, {120, 1}, {30, 6}, {30, 1}, {6, 1}};
};

class OrderTest : public NDArrayTests {
 public:
  sd::LongType expected[8] = {2, 3, 4, 1, 3, 0, 0, 102};
  sd::LongType test[8] = {2, 3, 4, 1, 3, 0, 0, 102};
};

class LeadingOnes : public NDArrayTests {
 public:
  sd::LongType shapeBufferF[16] = {4, 1, 1, 4, 4, 1, 1, 1, 4, 16384, 1, 102};  // shapes with data type DOUBLE
  sd::LongType shapeBufferC[16] = {4, 1, 1, 4, 4, 16, 16, 4, 1, 16384, 1, 99};
  int dimensionLength = 2;
  sd::LongType dimension[2] = {2, 3};
  sd::LongType tadAssertionC[10] = {3, 4, 4, 1, 4, 1, 16, 16384, 1, 99};
  sd::LongType tadCAssertionF[10] = {3, 4, 4, 1, 1, 4, 1, 16384, 1, 102};
};

TEST_F(LeadingOnes, OnesTest) {
  shape::TAD *cTad = new shape::TAD;
  cTad->init(shapeBufferC, dimension, dimensionLength);
  cTad->createTadOnlyShapeInfo();
  cTad->createOffsets();
  shape::TAD *fTad = new shape::TAD;
  fTad->init(shapeBufferF, dimension, dimensionLength);
  fTad->createTadOnlyShapeInfo();
  fTad->createOffsets();
  ASSERT_TRUE(arrsEquals(10, tadCAssertionF, fTad->tadOnlyShapeInfo));
  ASSERT_TRUE(arrsEquals(10, tadAssertionC, cTad->tadOnlyShapeInfo));

  delete cTad;
  delete fTad;
}

class NormalThreeFourFive : public NDArrayTests {
 public:
  sd::LongType assertionBuffer[8] = {2, 3, 4, 20, 5, 16384, 5, 99};
  sd::LongType inputShapeBuffer[10] = {3, 3, 4, 5, 20, 5, 1, 16384, 1, 99};
  int dimensionLength = 2;
  sd::LongType dimension[2] = {0, 1};
};

TEST_F(NormalThreeFourFive, DimensionTest) {
  shape::TAD *tad = new shape::TAD;
  tad->init(inputShapeBuffer, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  ASSERT_TRUE(arrsEquals(8, assertionBuffer, tad->tadOnlyShapeInfo));

  delete tad;
}

class DimensionWarning : public NDArrayTests {
 public:
  int dimensionLength = 2;
  sd::LongType dimensions[2] = {0, 1};
  sd::LongType shape[3] = {1, 5, 1};
  sd::LongType *shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 3, shape);

  ~DimensionWarning() { delete[] shapeBuffer; }
};

TEST_F(DimensionWarning, ShapeWarning) {
  shape::TAD *tad = new shape::TAD;
  tad->init(shapeBuffer, dimensions, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  delete tad;
}

class TadRank : public NDArrayTests {
  sd::LongType shapeBuffer[12] = {4, 2, 1, 3, 3, 9, 9, 3, 1, 0, 1, 99};
  int dimensionLength = 2;
  sd::LongType dimension[2] = {2, 3};
};

class TestRemoveIndex : public NDArrayTests {};

class TestReverseCopy : public NDArrayTests {};

class TestConcat : public NDArrayTests {};

class SliceVectorTest : public NDArrayTests {};

class SliceMatrixTest : public NDArrayTests {};

class SliceTensorTest : public NDArrayTests {};

class ElementWiseStrideTest : public NDArrayTests {
 public:
  sd::LongType shape[3] = {3, 4, 5};
  sd::LongType stride[2] = {20, 5};
  int elementWiseStrideAssertion = -1;
};

class PermuteTest : public NDArrayTests {};

class LengthPerSliceTest : public NDArrayTests {};

class ExpectedValuesTest : public NDArrayTests {
 public:
  sd::LongType mainShape[4] = {9, 7, 5, 3};
  sd::LongType testDimensions[3] = {0, 2, 3};
};

class BeginOneTadTest : public NDArrayTests {
 public:
  sd::LongType assertionShapeBuffer[8] = {2, 3, 5, 1, 3, 16384, 1, 102};
  sd::LongType inputShapeBuffer[10] = {3, 1, 3, 5, 1, 1, 3, 16384, 0, 102};
  int dimensionLength = 2;
  sd::LongType dimension[2] = {1, 2};
};

class FourDTest : public NDArrayTests {
  /**
   * INDArray array3d = Nd4j.ones(1, 10, 10);
array3d.sum(1);

INDArray array4d = Nd4j.ones(1, 10, 10, 10);
INDArray sum40 = array4d.sum(0);
   */
 public:
  sd::LongType threeDShape[3] = {1, 10, 10};
  sd::LongType fourDShape[4] = {1, 10, 10, 10};
  sd::LongType *threeDShapeBuffer = nullptr, *fourDShapeBuffer = nullptr;
  sd::LongType dimensionThree = 1;
  sd::LongType dimensionThreeTwo = 0;
  sd::LongType dimensionFour = 0;
  sd::LongType dimensionLength = 1;
  FourDTest() {
    threeDShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'f', 3, threeDShape);
    fourDShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'f', 4, fourDShape);
  }
  ~FourDTest() {
    if (threeDShapeBuffer != nullptr) delete[] threeDShapeBuffer;
    if (fourDShapeBuffer != nullptr) delete[] fourDShapeBuffer;
  }
};

TEST_F(FourDTest, ThreeDFourDTest) {
  shape::TAD *threeTadTwo = new shape::TAD;
  threeTadTwo->init(threeDShapeBuffer, &dimensionThreeTwo, dimensionLength);
  threeTadTwo->createTadOnlyShapeInfo();
  threeTadTwo->createOffsets();

  shape::TAD *threeTad = new shape::TAD;
  threeTad->init(threeDShapeBuffer, &dimensionThree, dimensionLength);
  threeTad->createTadOnlyShapeInfo();
  threeTad->createOffsets();

  shape::TAD *fourTad = new shape::TAD;
  fourTad->init(fourDShapeBuffer, &dimensionFour, dimensionLength);
  fourTad->createTadOnlyShapeInfo();
  fourTad->createOffsets();

  delete threeTadTwo;
  delete threeTad;
  delete fourTad;
}

class RowVectorOnesTest : public NDArrayTests {
 public:
  sd::LongType shapeBuffer[12] = {4, 4, 3, 1, 1, 3, 1, 1, 1, 8192, 1, 99};  // float32 type of shape
  float data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  sd::LongType assertionBuffer[10] = {3, 4, 1, 1, 3, 1, 1, 8192, 0, 99};
  int dimensionLength = 3;
  sd::LongType dimension[3] = {0, 2, 3};
};



class SixDTest : public NDArrayTests {
 public:
  sd::LongType inputShapeBuffer[16] = {6, 1, 1, 4,  4,  4,     4, 1,
                                       1, 1, 4, 16, 64, 16384, 1, 102};  // shape with double data type
  int dimensionLength = 2;
  sd::LongType dimension[2] = {2, 3};
  sd::LongType assertionShapeBuffer[8] = {2, 4, 4, 1, 4, 16384, 1, 102};  // also double typed shape
};

TEST_F(SixDTest, SixDWithOnes) {
  shape::TAD *tad = new shape::TAD;
  tad->init(inputShapeBuffer, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  // shape::printShapeInfoLinear(inputShapeBuffer);
  // shape::printShapeInfoLinear(tad->tadOnlyShapeInfo);
  //[2,1,1,1,1,0,1,97]
  ASSERT_TRUE(arrsEquals(8, assertionShapeBuffer, tad->tadOnlyShapeInfo));
  delete tad;
}

class TrailingTest : public NDArrayTests {
 public:
  sd::LongType inputShapeBuffer[12] = {4, 5, 5, 5, 1, 1, 5, 25, 125, 16384, 1, 102};
  int dimensionLength = 1;
  sd::LongType dimension[1] = {0};
  sd::LongType assertionShapeBuffer[8] = {2, 1, 5, 125, 1, 16384, 1, 102};
};

TEST_F(TrailingTest, TrailingTest2) {
  shape::TAD *tad = new shape::TAD;
  tad->init(inputShapeBuffer, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  //[2,1,1,1,1,0,1,97]
  ASSERT_TRUE(arrsEquals(8, assertionShapeBuffer, tad->tadOnlyShapeInfo));
  delete tad;
}

class ScalarTest : public NDArrayTests {
 public:
  sd::LongType inputShapeBuffer[12] = {3, 2, 3, 4, 12, 4, 1, 16384, 1, 99};
  int dimensionLength = 1;
  sd::LongType dimension[1] = {1};
  sd::LongType assertionShapeBuffer[8] = {2, 1, 1, 1, 1, 16384, 1, 99};
};
/*
TEST_F(ScalarTest,ScalarTest2) {
    shape::TAD *tad = new shape::TAD(inputShapeBuffer,dimension,dimensionLength);
    tad->createTadOnlyShapeInfo();
    tad ->createOffsets();
    //[2,1,1,1,1,0,1,97]
    shape::printShapeInfoLinear(tad->tadOnlyShapeInfo);
    ASSERT_TRUE(arrsEquals(8,assertionShapeBuffer,tad->tadOnlyShapeInfo));
}
*/

class ThreeTest : public NDArrayTests {
 public:
  sd::LongType inputShapeBuffer[10] = {3, 4, 3, 2, 6, 2, 1, 16384, 1, 99};
  int dimensionLength = 1;
  sd::LongType dimension[1] = {0};
  sd::LongType assertionShapeBuffer[8] = {2, 1, 4, 1, 6, 16384, 6, 99};
};

TEST_F(ThreeTest, ThreeTest) {
  shape::TAD *tad = new shape::TAD;
  tad->init(inputShapeBuffer, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  tad->createOffsets();
  //[2,1,1,1,1,0,1,97]
  ASSERT_TRUE(arrsEquals(8, assertionShapeBuffer, tad->tadOnlyShapeInfo));
  delete tad;
}

TEST_F(BeginOneTadTest, TadTest) {
  shape::TAD *tad = new shape::TAD;
  tad->init(inputShapeBuffer, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  auto tadShapeBuffer = tad->tadOnlyShapeInfo;
  // shape::printShapeInfoLinear(tadShapeBuffer);
  //[2,1,1,1,1,0,1,97]
  ASSERT_TRUE(arrsEquals(8, assertionShapeBuffer, tadShapeBuffer));

  delete tad;
}



TEST_F(LabelTest, LabelTad) {
  shape::TAD *tad = new shape::TAD;
  tad->init(shapeInfo, dimension, dimensionLength);
  tad->createTadOnlyShapeInfo();
  auto tadShapeInfo = tad->tadOnlyShapeInfo;
  ASSERT_TRUE(arrsEquals(8, tadShapeInfoAssert, tadShapeInfo));

  delete tad;
}

TEST_F(ExpectedValuesTest, TadTest) {
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, mainShape);
  shape::TAD *tad = new shape::TAD;
  tad->init(shapeBuffer, testDimensions, 3);
  tad->createTadOnlyShapeInfo();
  auto shapeInfo = tad->tadOnlyShapeInfo;

  delete tad;
  delete[] shapeBuffer;
}

TEST_F(OrderTest, testOrder) {
  int rank = shape::rank(expected);
  auto expectedShape = shape::shapeOf(expected);
  auto expectedStride = shape::stride(expected);
  int realOrder = shape::getOrder(rank, expectedShape, expectedStride, 1);
  int expectedOrder = 102;
  ASSERT_EQ(expectedOrder, realOrder);
}

TEST_F(ThreeDTest, TensorAlongDimensionTest) {
  sd::LongType dimension[2] = {0, 2};
  sd::LongType tadShapeAssertion[2] = {3, 5};
  sd::LongType strideAssertion[2] = {20, 1};
  shape::TAD *tad = new shape::TAD;
  tad->init(0, this->shapeBuffer, dimension, 2);
  tad->createTadOnlyShapeInfo();
  auto shapeBufferTest = tad->tadOnlyShapeInfo;
  auto shapeTest = shape::shapeOf(shapeBufferTest);
  auto strideTest = shape::stride(shapeBufferTest);
  ASSERT_TRUE(arrsEquals(2, tadShapeAssertion, shapeTest));
  ASSERT_TRUE(arrsEquals(2, strideAssertion, strideTest));
  delete tad;
}

TEST_F(NumTadTests, TadTest) {
  auto shape = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 3, this->shape);
  shape::TAD *tad = new shape::TAD;
  tad->init(shape, &dimension, 1);
  int numTads = shape::tensorsAlongDimension(shape, &dimension, 1);
  ASSERT_EQ(20, numTads);
  delete[] shape;
  delete tad;
}

TEST_F(TADStall, TestStall) {
  auto shapeInfo = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, shape);
  shape::TAD *tad = new shape::TAD;
  tad->init(0, shapeInfo, this->dimensions, 3);
  tad->createTadOnlyShapeInfo();
  sd::LongType *test = tad->tadOnlyShapeInfo;

  delete[] shapeInfo;
  delete tad;
}

TEST_F(LengthPerSliceTest, TestLengthPerSlice) {
  sd::LongType firstShape[2] = {5, 3};
  int lengthPerSliceAssertionFirst = 3;
  sd::LongType firstDimension = 0;
  int lengthPerSliceTest = shape::lengthPerSlice(2, firstShape, &firstDimension, 1);
  ASSERT_EQ(lengthPerSliceAssertionFirst, lengthPerSliceTest);
}

TEST_F(PermuteTest, PermuteShapeBufferTest) {
  sd::LongType permuteOrder[4] = {3, 2, 1, 0};
  sd::LongType normalOrder[4] = {0, 1, 2, 3};
  sd::LongType shapeToPermute[4] = {5, 3, 2, 6};
  sd::LongType permutedOrder[4] = {6, 2, 3, 5};
  auto shapeBufferOriginal = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, shapeToPermute);
  auto assertionShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, shapeToPermute);
  shape::permuteShapeBufferInPlace(shapeBufferOriginal, normalOrder, shapeBufferOriginal);
  EXPECT_TRUE(arrsEquals(4, assertionShapeBuffer, shapeBufferOriginal));

  auto backwardsAssertion = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, permutedOrder);
  auto permuted = shape::permuteShapeBuffer(assertionShapeBuffer, permuteOrder);
  EXPECT_TRUE(arrsEquals(4, backwardsAssertion, permuted));

  delete[] permuted;
  delete[] backwardsAssertion;
  delete[] shapeBufferOriginal;
  delete[] assertionShapeBuffer;
}

TEST_F(ElementWiseStrideTest, ElementWiseStrideTest) {}

TEST_F(SliceVectorTest, RowColumnVectorTest) {
  sd::LongType rowVectorShape[2] = {1, 5};
  auto rowVectorShapeInfo = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, rowVectorShape);
  sd::LongType colVectorShape[2] = {5, 1};
  auto colVectorShapeInfo = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, colVectorShape);
  sd::LongType *sliceRow = shape::sliceOfShapeBuffer(0, rowVectorShapeInfo);
  EXPECT_TRUE(arrsEquals(2, rowVectorShapeInfo, sliceRow));
  sd::LongType *scalarSliceInfo = shape::createScalarShapeInfo();
  sd::LongType *scalarColumnAssertion = shape::createScalarShapeInfo();
  scalarColumnAssertion[shape::shapeInfoLength(2) - 3] = 1;
  sd::LongType *scalarColumnTest = shape::sliceOfShapeBuffer(1L, colVectorShapeInfo);
  EXPECT_TRUE(arrsEquals(2, scalarColumnAssertion, scalarColumnTest));

  delete[] scalarColumnTest;
  delete[] scalarColumnAssertion;
  delete[] scalarSliceInfo;
  delete[] sliceRow;
  delete[] rowVectorShapeInfo;
  delete[] colVectorShapeInfo;
}

TEST_F(SliceTensorTest, TestSlice) {
  sd::LongType shape[3] = {3, 3, 2};
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 3, shape);
  sd::LongType sliceShape[2] = {3, 2};
  auto sliceShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, sliceShape);
  sd::LongType *testSlice = shape::sliceOfShapeBuffer(0, shapeBuffer);
  EXPECT_TRUE(arrsEquals(2, sliceShapeBuffer, testSlice));
  delete[] testSlice;
  delete[] shapeBuffer;
  delete[] sliceShapeBuffer;
}

TEST_F(SliceMatrixTest, TestSlice) {
  sd::LongType shape[2] = {3, 2};
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, shape);
  sd::LongType sliceShape[2] = {1, 2};
  auto sliceShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, sliceShape);
  sd::LongType *testSlice = shape::sliceOfShapeBuffer(0, shapeBuffer);
  EXPECT_TRUE(arrsEquals(2, sliceShapeBuffer, testSlice));
  delete[] testSlice;
  delete[] shapeBuffer;
  delete[] sliceShapeBuffer;
}

TEST_F(TestConcat, ConcatTest) {
  sd::LongType firstArr[2] = {1, 2};
  sd::LongType secondConcat[2] = {3, 4};
  sd::LongType concatAssertion[4] = {1, 2, 3, 4};
  sd::LongType *concatTest = shape::concat(firstArr, 2, secondConcat, 2);
  EXPECT_TRUE(arrsEquals(4, concatAssertion, concatTest));
  delete[] concatTest;
}

TEST_F(TestReverseCopy, ReverseCopyTest) {
  sd::LongType toCopy[5] = {0, 1, 2, 3, 4};
  sd::LongType reverseAssertion[5] = {4, 3, 2, 1, 0};
  sd::LongType *reverseCopyTest = shape::reverseCopy(toCopy, 5);
  EXPECT_TRUE(arrsEquals(5, reverseAssertion, reverseCopyTest));
  delete[] reverseCopyTest;
}

TEST_F(TestRemoveIndex, Remove) {
  sd::LongType input[5] = {0, 1, 2, 3, 4};
  sd::LongType indexesToRemove[3] = {0, 1, 2};
  sd::LongType indexesToRemoveAssertion[2] = {3, 4};
  sd::LongType *indexesToRemoveTest =
      shape::removeIndex<sd::LongType>(input, indexesToRemove, (sd::LongType)5, (sd::LongType)3);
  EXPECT_TRUE(arrsEquals(2, indexesToRemoveAssertion, indexesToRemoveTest));
  delete[] indexesToRemoveTest;
}

TEST_F(TensorTwoFromFourDDimTest, TadTwoFromFourDimTest) {
  // Along dimension 0,1: expect matrix with shape [rows,cols]
  // Along dimension 0,2: expect matrix with shape [rows,dim2]
  // Along dimension 0,3: expect matrix with shape [rows,dim3]
  // Along dimension 1,2: expect matrix with shape [cols,dim2]
  // Along dimension 1,3: expect matrix with shape [cols,dim3]
  // Along dimension 2,3: expect matrix with shape [dim2,dim3]
  auto baseShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 4, shape);
  for (int i = 0; i < 3; i++) {
    sd::LongType *dimArr = dims[i];
    sd::LongType *expectedShape = expectedShapes[i];
    shape::TAD *tad = new shape::TAD;
    tad->init(baseShapeBuffer, dimArr, dimensionLength);
    auto expectedShapeBuffer =
        sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', dimensionLength, expectedShape);
    tad->createTadOnlyShapeInfo();
    sd::LongType *testShapeBuffer = tad->tadOnlyShapeInfo;
    EXPECT_TRUE(arrsEquals(shape::rank(expectedShapeBuffer), expectedShape, shape::shapeOf(testShapeBuffer)));
    EXPECT_TRUE(arrsEquals(shape::rank(expectedShapeBuffer), expectedStrides[i], shape::stride(testShapeBuffer)));

    delete[] expectedShapeBuffer;
    delete tad;
  }

  delete[] baseShapeBuffer;
}

TEST_F(TensorTwoDimTest, TadTwoDimTest) {
  // Along dimension 0,1: expect matrix with shape [rows,cols]
  // Along dimension 0,2: expect matrix with shape [rows,dim2]
  // Along dimension 1,2: expect matrix with shape [cols,dim2]
  auto baseShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 3, shape);

  for (int i = 0; i < 3; i++) {
    sd::LongType *dimArr = dims[i];
    sd::LongType *expectedShape = expectedShapes[i];
    shape::TAD *tad = new shape::TAD;
    tad->init(baseShapeBuffer, dimArr, dimensionLength);
    auto expectedShapeBuffer =
        sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', dimensionLength, expectedShape);
    tad->createTadOnlyShapeInfo();
    sd::LongType *testShapeBuffer = tad->tadOnlyShapeInfo;
    sd::LongType *expectedStride = expectedStrides[i];
    sd::LongType *testShape = shape::shapeOf(testShapeBuffer);
    sd::LongType *testStride = shape::stride(testShapeBuffer);
    EXPECT_TRUE(arrsEquals(shape::rank(expectedShapeBuffer), expectedShape, testShape));
    EXPECT_TRUE(arrsEquals(shape::rank(testShapeBuffer), expectedStride, testStride));

    delete[] expectedShapeBuffer;
    delete tad;
  }

  delete[] baseShapeBuffer;
}

TEST_F(TensorOneDimTest, TadDimensionsForTensor) {
  sd::LongType shape[3] = {rows, cols, dim2};
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', rank, shape);

  for (int i = 0; i < rank; i++) {
    // Along dimension 0: expect row vector with length 'dims[i]'
    shape::TAD *zero = new shape::TAD;
    zero->init(shapeBuffer, &dims[i], 1);
    zero->createTadOnlyShapeInfo();
    sd::LongType *testDimZeroShapeBuffer = zero->tadOnlyShapeInfo;
    sd::LongType *testShape = shape::shapeOf(testDimZeroShapeBuffer);
    sd::LongType *testStride = shape::stride(testDimZeroShapeBuffer);
    EXPECT_TRUE(arrsEquals(2, expectedShapes[i], testShape));
    EXPECT_TRUE(arrsEquals(2, expectedStrides[i], testStride));

    delete zero;
  }

  delete[] shapeBuffer;
}

TEST_F(MatrixTest, TadDimensionsForMatrix) {
  sd::LongType shape[2] = {rows, cols};
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', rank, shape);

  shape::TAD *dimZero = new shape::TAD;
  dimZero->init(shapeBuffer, &dims[0], 1);
  shape::TAD *dimOne = new shape::TAD;
  dimOne->init(shapeBuffer, &dims[1], 1);
  // Along dimension 0: expect row vector with length 'rows'
  sd::LongType rowVectorShape[2] = {1, rows};
  auto expectedDimZeroShape = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, rowVectorShape);
  dimZero->createTadOnlyShapeInfo();
  sd::LongType *testDimZero = dimZero->tadOnlyShapeInfo;
  EXPECT_TRUE(arrsEquals(2, expectedShapes[0], shape::shapeOf(testDimZero)));
  EXPECT_TRUE(arrsEquals(2, expectedStrides[0], shape::stride(testDimZero)));

  delete[] expectedDimZeroShape;
  // Along dimension 1: expect row vector with length 'cols'
  sd::LongType rowVectorColShape[2]{1, cols};
  auto expectedDimOneShape = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, rowVectorColShape);
  dimOne->createTadOnlyShapeInfo();
  sd::LongType *testDimOneShape = dimOne->tadOnlyShapeInfo;
  EXPECT_TRUE(arrsEquals(2, expectedShapes[1], shape::shapeOf(testDimOneShape)));
  EXPECT_TRUE(arrsEquals(2, expectedStrides[1], shape::stride(testDimOneShape)));

  delete[] expectedDimOneShape;
  delete dimOne;
  delete dimZero;
  delete[] shapeBuffer;
}

TEST_F(VectorTest, VectorTadShape) {
  sd::LongType rowVector[2] = {2, 2};
  auto rowBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, rowVector);
  sd::LongType rowDimension = 1;

  sd::LongType columnVector[2] = {2, 2};
  auto colShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, columnVector);
  sd::LongType colDimension = 0;

  shape::TAD *rowTad = new shape::TAD;
  rowTad->init(rowBuffer, &rowDimension, 1);
  rowTad->createTadOnlyShapeInfo();
  sd::LongType *rowTadShapeBuffer = rowTad->tadOnlyShapeInfo;
  sd::LongType *rowTadShape = shape::shapeOf(rowTadShapeBuffer);
  shape::TAD *colTad = new shape::TAD;
  colTad->init(colShapeBuffer, &colDimension, 1);
  colTad->createTadOnlyShapeInfo();
  sd::LongType *colTadShapeBuffer = colTad->tadOnlyShapeInfo;
  sd::LongType *colTadShape = shape::shapeOf(colTadShapeBuffer);
  sd::LongType assertionShape[2] = {1, 2};
  sd::LongType assertionStride[2] = {1, 1};
  EXPECT_TRUE(arrsEquals(2, assertionShape, rowTadShape));
  EXPECT_TRUE(arrsEquals(2, assertionStride, shape::stride(rowTadShapeBuffer)));
  EXPECT_TRUE(arrsEquals(2, assertionShape, colTadShape));

  delete[] rowBuffer;
  delete[] colShapeBuffer;
  delete rowTad;
  delete colTad;
}

TEST_F(ShapeTest, IsVector) { ASSERT_TRUE(shape::isVector(vectorShape, 2)); }

TEST_F(VectorTest, LinspaceCombinationTest) {
  int rows = 3;
  int cols = 4;
  int len = rows * cols;
  double *linspaced = linspace<double>(1, rows * cols, len);
  sd::LongType shape[2] = {rows, cols};
  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, shape);

  delete[] shapeBuffer;
  delete[] linspaced;
}
