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
#include <helpers/MmulHelper.h>

#include <memory>

#include "testlayers.h"

using namespace sd;

//////////////////////////////////////////////////////////////////////
class NDArrayTest : public NDArrayTests {
 public:
  int alpha = 0;

  LongType *cShape = new LongType[8]{2, 2, 2, 2, 1, 8192, 1, 99};
  LongType *fShape = new LongType[8]{2, 2, 2, 1, 2, 8192, 1, 102};

  float arr1[6] = {1, 2, 3, 4, 5, 6};
  LongType shape1[8] = {2, 2, 3, 3, 1, 8192, 1, 99};
  float arr2[48] = {1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6,
                    1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6};
  LongType shape2[10] = {3, 2, 4, 6, 24, 6, 1, 8192, 1, 99};
  const std::vector<LongType> tileShape1 = {2, 2, 2};

  ~NDArrayTest() {
    delete[] cShape;
    delete[] fShape;
  }
};

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestDup1) {
  NDArray array(arr1, shape1);

  auto arrC = new NDArray(array.dup('c'));
  auto arrF = new NDArray(array.dup('f'));

  ASSERT_TRUE(array.equalsTo(arrF));
  ASSERT_TRUE(array.equalsTo(arrC));

  ASSERT_TRUE(arrF->equalsTo(arrC));

  delete arrC;
  delete arrF;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, AssignScalar1) {
  auto array = NDArrayFactory::create_<float>('c', {1, 10});

  array->assign(2.0f);

  for (int i = 0; i < array->lengthOf(); i++) {
    ASSERT_EQ(2.0f, array->e<float>(i));
  }

  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, NDArrayOrder1) {
  // original part
  auto c = new float[4]{1, 2, 3, 4};

  // expected part
  auto f = new float[4]{1, 3, 2, 4};

  auto arrayC = new NDArray(c, cShape);
  auto arrayF = new NDArray(arrayC->dup('f'));
  auto arrayC2 = new NDArray(arrayF->dup('c'));

  arrayF->syncToHost();
  arrayC2->syncToHost();

  ASSERT_EQ('c', arrayC->ordering());
  ASSERT_EQ('f', arrayF->ordering());
  ASSERT_EQ('c', arrayC2->ordering());

  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(f[i], arrayF->bufferAsT<float>()[i], 1e-5f);
  }

  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(fShape[i], arrayF->shapeInfo()[i]);
  }

  for (int i = 0; i < 4; i++) {
    ASSERT_NEAR(c[i], arrayC2->bufferAsT<float>()[i], 1e-5f);
  }

  for (int i = 0; i < 8; i++) {
    ASSERT_EQ(cShape[i], arrayC2->shapeInfo()[i]);
  }

  delete[] c;
  delete[] f;
  delete arrayC;
  delete arrayF;
  delete arrayC2;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestGetScalar1) {
  auto c = new float[4]{1, 2, 3, 4};
  auto cShape = new LongType[8]{2, 2, 2, 2, 1, 8192, 1, 99};

  auto arrayC = new NDArray(c, cShape);

  ASSERT_NEAR(3.0f, arrayC->e<float>(1, 0), 1e-5f);
  ASSERT_NEAR(4.0f, arrayC->e<float>(1, 1), 1e-5f);

  auto arrayF = new NDArray(arrayC->dup('f'));

  ASSERT_NEAR(3.0f, arrayF->e<float>(1, 0), 1e-5f);
  ASSERT_NEAR(4.0f, arrayF->e<float>(1, 1), 1e-5f);

  arrayF->p(1, 0, 7.0f);
  ASSERT_NEAR(7.0f, arrayF->e<float>(1, 0), 1e-5f);

  arrayC->p(1, 1, 9.0f);
  ASSERT_NEAR(9.0f, arrayC->e<float>(1, 1), 1e-5f);

  delete[] c;
  delete[] cShape;

  delete arrayC;
  delete arrayF;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, EqualityTest1) {
  auto arrayA = NDArrayFactory::create_<float>('f', {3, 5});
  auto arrayB = NDArrayFactory::create_<float>('f', {3, 5});
  auto arrayC = NDArrayFactory::create_<float>('f', {3, 5});

  auto arrayD = NDArrayFactory::create_<float>('f', {2, 4});
  auto arrayE = NDArrayFactory::create_<float>('f', {1, 15});

  for (int i = 0; i < arrayA->rows(); i++) {
    for (int k = 0; k < arrayA->columns(); k++) {
      arrayA->p(i, k, (float)i);
    }
  }

  for (int i = 0; i < arrayB->rows(); i++) {
    for (int k = 0; k < arrayB->columns(); k++) {
      arrayB->p(i, k, (float)i);
    }
  }

  for (int i = 0; i < arrayC->rows(); i++) {
    for (int k = 0; k < arrayC->columns(); k++) {
      arrayC->p(i, k, (float)i + 1);
    }
  }

  ASSERT_TRUE(arrayA->equalsTo(arrayB, 1e-5));
  ASSERT_FALSE(arrayC->equalsTo(arrayB, 1e-5));
  ASSERT_FALSE(arrayD->equalsTo(arrayB, 1e-5));
  ASSERT_FALSE(arrayE->equalsTo(arrayB, 1e-5));

  delete arrayA;
  delete arrayB;
  delete arrayC;
  delete arrayD;
  delete arrayE;
}

TEST_F(NDArrayTest, TestTad1) {
  auto array = NDArrayFactory::create_<float>('c', {3, 3});

  auto row2 = (*array)(1, {0});

  ASSERT_TRUE(row2.isView());
  ASSERT_EQ(3, row2.lengthOf());

  row2.assign(1.0);

  ASSERT_NEAR(3.0f, array->sumNumber().e<float>(0), 1e-5);
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad2) {
  auto array = NDArrayFactory::create_<float>('c', {3, 3});

  ASSERT_EQ(3, array->tensorsAlongDimension({1}));

  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTad3) {
  auto array = NDArrayFactory::create_<float>('c', {4, 3});

  auto row2 = (*array)(1, {0});

  ASSERT_TRUE(row2.isView());
  ASSERT_EQ(3, row2.lengthOf());
  delete array;
}

TEST_F(NDArrayTest, TestPermuteReshape1) {
  NDArray array('c', {2, 2, 5, 5}, FLOAT32);
  int pShape[] = {4, 2, 5, 5, 2, 25, 5, 1, 50, 8192, 0, 99};
  int rShape[] = {3, 2, 25, 2, 25, 1, 50, 8192, 0, 99};

  array.permutei({1, 2, 3, 0});

  for (int e = 0; e < shape::shapeInfoLength(array.shapeInfo()); e++) ASSERT_EQ(pShape[e], array.shapeInfo()[e]);

  array.reshapei('c', {2, 25, 2});

  for (int e = 0; e < shape::shapeInfoLength(array.shapeInfo()); e++) ASSERT_EQ(rShape[e], array.shapeInfo()[e]);
}

TEST_F(NDArrayTest, TestPermuteReshape2) {
  auto array = NDArrayFactory::create<float>('c', {2, 2, 5, 5, 6, 6});
  int pShape[] = {6, 2, 2, 6, 6, 5, 5, 900, 1800, 6, 1, 180, 36, 8192, 0, 99};
  int rShape[] = {3, 2, 72, 25, 1800, 25, 1, 8192, 1, 99};


  array.permutei({1, 0, 4, 5, 2, 3});


  auto aShape = array.shapeInfo();

  for (int e = 0; e < shape::shapeInfoLength(array.shapeInfo()); e++) ASSERT_EQ(pShape[e], aShape[e]);

  array.reshapei('c', {2, 72, 25});

  for (int e = 0; e < shape::shapeInfoLength(array.shapeInfo()); e++) ASSERT_EQ(rShape[e], array.shapeInfo()[e]);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestRepeat1) {
  auto eBuffer = new float[8]{1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
  auto eShape = new LongType[8]{2, 4, 2, 2, 1, 8192, 1, 99};
  NDArray array('c', {2, 2}, FLOAT32);
  auto exp = new NDArray(eBuffer, eShape);
  for (int e = 0; e < array.lengthOf(); e++) array.p(e, e + 1);


  auto rep = array.repeat(0, {2});

  ASSERT_EQ(4, rep.sizeAt(0));
  ASSERT_EQ(2, rep.sizeAt(1));

  ASSERT_TRUE(exp->equalsTo(rep));

  delete[] eBuffer;
  delete[] eShape;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestRepeat2) {
  auto eBuffer = new float[8]{1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0};
  auto eShape = new LongType[8]{2, 4, 2, 2, 1, 8192, 1, 99};
  auto array = NDArrayFactory::create_<float>('c', {2, 2});
  auto exp = new NDArray(eBuffer, eShape);
  for (int e = 0; e < array->lengthOf(); e++) array->p(e, e + 1);


  auto rep = new NDArray(exp->dup());
  rep->assign(0.);
  array->repeat(0, {2}, *rep);

  ASSERT_EQ(4, rep->sizeAt(0));
  ASSERT_EQ(2, rep->sizeAt(1));


  ASSERT_TRUE(exp->equalsTo(rep));

  delete[] eBuffer;
  delete[] eShape;
  delete array;
  delete exp;
  delete rep;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestIndexedPut1) {
  auto array = NDArrayFactory::create_<float>('f', {3, 3});

  array->p(4, 1.0f);
  ASSERT_EQ(1.0f, array->e<float>(4));

  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSum1) {
  // sd::LongType *cShape = new sd::LongType[8]{2, 2, 2, 2, 1, 8192, 1, 99};
  float *c = new float[4]{1, 2, 3, 4};

  auto array = new NDArray(c, cShape);

  ASSERT_EQ(10.0f, array->sumNumber().e<float>(0));
  ASSERT_EQ(2.5f, array->meanNumber().e<float>(0));

  delete[] c;
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestAddiRowVector) {
  float *c = new float[4]{1, 2, 3, 4};
  float *e = new float[4]{2, 3, 4, 5};

  auto array = new NDArray(c, cShape);
  auto row = NDArrayFactory::create_<float>('c', {1, 2});
  auto exp = new NDArray(e, cShape);
  row->assign(1.0f);

  array->addiRowVector(*row);

  ASSERT_TRUE(exp->equalsTo(array));

  delete[] c;
  delete[] e;

  delete array;
  delete row;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestAddiColumnVector) {
  float arr1[] = {1, 2, 3, 4};
  float arr2[] = {5, 6};
  float arr3[] = {6, 7, 9, 10};
  LongType shape1[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType shape2[] = {2, 2, 1, 1, 1, 8192, 1, 99};
  NDArray matrix(arr1, shape1);
  NDArray column(arr2, shape2);
  NDArray exp(arr3, shape1);

  matrix.addiColumnVector(column);
  ASSERT_TRUE(exp.isSameShapeStrict(matrix));
  ASSERT_TRUE(exp.equalsTo(&matrix));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMuliColumnVector) {
  float arr1[] = {1, 2, 3, 4};
  float arr2[] = {5, 6};
  float arr3[] = {5, 10, 18, 24};
  LongType shape1[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType shape2[] = {2, 2, 1, 1, 1, 8192, 1, 99};
  NDArray matrix(arr1, shape1);
  NDArray column(arr2, shape2);
  NDArray exp(arr3, shape1);

  matrix.muliColumnVector(column);

  ASSERT_TRUE(exp.isSameShapeStrict(matrix));
  ASSERT_TRUE(exp.equalsTo(&matrix));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test3D_1) {
  auto arrayC = NDArrayFactory::create_<double>('c', {2, 5, 10});
  auto arrayF = NDArrayFactory::create_<double>('f', {2, 5, 10});

  ASSERT_EQ(100, arrayC->lengthOf());
  ASSERT_EQ(100, arrayF->lengthOf());

  ASSERT_EQ('c', arrayC->ordering());
  ASSERT_EQ('f', arrayF->ordering());

  delete arrayC;
  delete arrayF;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose1) {
  auto arrayC = NDArrayFactory::create_<double>('c', {2, 5, 10});

  auto expC = new LongType[10]{3, 2, 5, 10, 50, 10, 1, 16384, 1, 99};
  auto expT = new LongType[10]{3, 10, 5, 2, 1, 10, 50, 16384, 1, 102};

  auto arrayT = arrayC->transpose();

  for (int e = 0; e < arrayC->rankOf(); e++) {
    ASSERT_EQ(shape::shapeOf(expC)[e], arrayC->sizeAt(e));
    ASSERT_EQ(shape::shapeOf(expT)[e], arrayT.sizeAt(e));
  }

  delete arrayC;
  delete[] expC;
  delete[] expT;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTranspose2) {
  auto arrayC = NDArrayFactory::create_<double>('c', {2, 5, 10});

  auto expC = new LongType[10]{3, 2, 5, 10, 50, 10, 1, 16384, 1, 99};
  auto expT = new LongType[10]{3, 10, 5, 2, 1, 10, 50, 16384, 1, 102};

  arrayC->transposei();

  for (int e = 0; e < arrayC->rankOf(); e++) {
    ASSERT_EQ(shape::shapeOf(expT)[e], arrayC->sizeAt(e));
  }

  delete arrayC;
  delete[] expC;
  delete[] expT;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceAlongDimension1) {
  NDArray array('c', {2, 2}, {1, 2, 3, 4}, FLOAT32);

  std::vector<LongType> zero = {0};
  auto res = array.reduceAlongDimension(reduce::Sum,&zero);

  ASSERT_EQ(2, res.lengthOf());

  ASSERT_EQ(4.0f, res.e<float>(0));
  ASSERT_EQ(6.0f, res.e<float>(1));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceAlongDimension2) {
  float *c = new float[4]{1, 2, 3, 4};
  auto array = new NDArray(c, cShape);
  std::vector<LongType> one = {1};
  auto res = array->reduceAlongDimension(reduce::Sum,&one);

  ASSERT_EQ(2, res.lengthOf());

  ASSERT_EQ(3.0f, res.e<float>(0));
  ASSERT_EQ(7.0f, res.e<float>(1));

  delete[] c;
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTransform1) {
  float *c = new float[4]{-1, -2, -3, -4};
  auto array = new NDArray(c, cShape);

  float *e = new float[4]{1, 2, 3, 4};
  auto exp = new NDArray(e, cShape);

  array->applyTransform(transform::Abs, array);

  ASSERT_TRUE(exp->equalsTo(array));

  delete[] c;
  delete array;
  delete[] e;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar1) {
  float *c = new float[4]{-1, -2, -3, -4};
  auto array = new NDArray(c, cShape);

  ASSERT_EQ(-4, array->reduceNumber(reduce::Min, nullptr).e<float>(0));

  delete[] c;
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar2) {
  float *c = new float[4]{-1, -2, -3, -4};
  auto array = new NDArray(c, cShape);

  ASSERT_EQ(-10, array->reduceNumber(reduce::Sum, nullptr).e<float>(0));

  delete[] c;
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestReduceScalar3) {
  auto array = new NDArray(arr1, shape1);

  ASSERT_EQ(21, array->reduceNumber(reduce::Sum, nullptr).e<float>(0));

  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestApplyTransform1) {
  float *c = new float[4]{-1, -2, -3, -4};
  auto array = new NDArray(c, cShape);

  float *e = new float[4]{1, 2, 3, 4};
  auto exp = new NDArray(e, cShape);

  array->applyTransform(transform::Abs, array);

  ASSERT_TRUE(exp->equalsTo(array));

  delete[] c;
  delete array;

  delete[] e;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVectors1) {
  float *c = new float[4]{-1, -2, -3, -4};
  auto array = new NDArray(c, cShape);

  auto vecShape = array->getShapeInfoAsVector();
  auto vecBuffer = array->getBufferAsVector<float>();

  ASSERT_EQ(8, vecShape.size());
  ASSERT_EQ(4, vecBuffer.size());

  for (int e = 0; e < vecBuffer.size(); e++) {
    ASSERT_NEAR(c[e], vecBuffer[e], 1e-5);
  }

  for (int e = 0; e < vecShape.size(); e++) {
    ASSERT_EQ(cShape[e], vecShape[e]);
  }

  delete[] c;
  delete array;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks1) {
  auto array = NDArrayFactory::create<float>('c', {1, 5});

  ASSERT_FALSE(array.isMatrix());
  ASSERT_FALSE(array.isScalar());
  ASSERT_TRUE(array.isVector());
  ASSERT_FALSE(array.isColumnVector());
  ASSERT_TRUE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks2) {
  auto array = NDArrayFactory::create<float>('c', {5, 5});

  ASSERT_TRUE(array.isMatrix());
  ASSERT_FALSE(array.isScalar());
  ASSERT_FALSE(array.isVector());
  ASSERT_FALSE(array.isColumnVector());
  ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks3) {
  auto array = NDArrayFactory::create<float>('c', {5, 1});

  ASSERT_FALSE(array.isMatrix());
  ASSERT_FALSE(array.isScalar());
  ASSERT_TRUE(array.isVector());
  ASSERT_TRUE(array.isColumnVector());
  ASSERT_FALSE(array.isRowVector());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks4) {
  auto array = NDArrayFactory::create<float>('c', {1, 1});

  ASSERT_FALSE(array.isMatrix());
  ASSERT_FALSE(array.isVector());
  ASSERT_FALSE(array.isColumnVector());
  ASSERT_FALSE(array.isRowVector());
  ASSERT_TRUE(array.isScalar());
}

TEST_F(NDArrayTest, TestReductionAny1) {
  auto array = NDArrayFactory::create<float>('c', {2, 2});
  array.p(0, 1.0f);
  array.p(1, 1.0f);
  array.p(2, 0.0f);
  array.p(3, 0.0f);
  array.syncToDevice();

  std::vector<LongType> zero = {0};
  std::vector<LongType> one = {1};

  auto result0 = array.reduceAlongDimension(reduce::Any,&zero);

  ASSERT_EQ(2, result0.lengthOf());

  ASSERT_NEAR(1.0f, result0.e<float>(0), 1e-5f);
  ASSERT_NEAR(1.0f, result0.e<float>(1), 1e-5f);

  auto result1 = array.reduceAlongDimension(reduce::Any,&one);

  ASSERT_EQ(2, result1.lengthOf());

  ASSERT_NEAR(1.0f, result1.e<float>(0), 1e-5f);
  ASSERT_NEAR(0.0f, result1.e<float>(1), 1e-5f);
}

TEST_F(NDArrayTest, TestReductionAll1) {
  auto array = NDArrayFactory::create<float>('c', {2, 2});
  array.p(0, 1.0f);
  array.p(1, 1.0f);
  array.p(2, 0.0f);
  array.p(3, 0.0f);

  //create vectors of sd::LongType containing 0 and 1
  std::vector<LongType> zero = {0};
  std::vector<LongType> one = {1};


  auto result0 = array.reduceAlongDimension(reduce::All, &zero);
  auto result1 = array.reduceAlongDimension(reduce::All, &one);

  ASSERT_EQ(2, result0.lengthOf());
  ASSERT_EQ(2, result1.lengthOf());

  ASSERT_FALSE(result0.e<bool>(0));
  ASSERT_FALSE(result0.e<bool>(1));

  ASSERT_TRUE(result1.e<bool>(0));
  ASSERT_FALSE(result1.e<bool>(1));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestChecks5) {
  auto array = NDArrayFactory::create<float>('c', {5, 5, 5});

  ASSERT_FALSE(array.isMatrix());
  ASSERT_FALSE(array.isVector());
  ASSERT_FALSE(array.isColumnVector());
  ASSERT_FALSE(array.isRowVector());
  ASSERT_FALSE(array.isScalar());
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile1) {


  NDArray array1(arr1, shape1);  // {2,3}
  NDArray array2(arr2, shape2);  // {2,4,6}
  auto expA = new NDArray(array1.dup('c'));

  auto tiled = array1.tile(tileShape1);
  ASSERT_TRUE(tiled.isSameShape(&array2));
  ASSERT_TRUE(tiled.equalsTo(&array2));

  ASSERT_TRUE(expA->isSameShape(&array1));
  ASSERT_TRUE(expA->equalsTo(&array1));

  delete expA;
}

TEST_F(NDArrayTest, TestTile2) {
  NDArray array1(arr1, shape1);
  NDArray array2(arr2, shape2);

  auto tiled = array1.tile(tileShape1);

  ASSERT_TRUE(tiled.isSameShape(&array2));
  ASSERT_TRUE(tiled.equalsTo(&array2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile3) {
  NDArray array1(arr1, shape1);
  NDArray array2(arr2, shape2);

  array1.tilei(tileShape1);

  ASSERT_TRUE(array1.isSameShapeStrict(array2));
  ASSERT_TRUE(array1.equalsTo(&array2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile4) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float expBuff[] = {1.f, 2.f, 1.f, 2.f, 3.f, 4.f, 3.f, 4.f, 5.f, 6.f, 5.f, 6.f};

  auto x = NDArrayFactory::create<float>(xBuff, 'c', {3, 1, 2});
  auto exp = NDArrayFactory::create<float>(expBuff, 'c', {3, 2, 2});

  auto result = x.tile({2, 1});

  ASSERT_TRUE(result.isSameShapeStrict(exp));
  ASSERT_TRUE(result.equalsTo(&exp));
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile5) {
  float xBuff[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float expBuff[] = {1.f, 2.f, 3.f, 4.f, 1.f, 2.f,  3.f,  4.f,  5.f, 6.f,  7.f,  8.f,
                     5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 9.f, 10.f, 11.f, 12.f};

  auto x = NDArrayFactory::create<float>(xBuff, 'c', {3, 2, 2});
  auto exp = NDArrayFactory::create<float>(expBuff, 'c', {3, 4, 2});

  auto result = x.tile({2, 1});

  ASSERT_TRUE(result.isSameShapeStrict(exp));
  ASSERT_TRUE(result.equalsTo(&exp));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTile6) {
  double expBuff[] = {10., 11., 10., 11., 10., 11., 10., 11., 12., 13., 12., 13.,
                      12., 13., 12., 13., 14., 15., 14., 15., 14., 15., 14., 15.};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);

  auto result = x.tile({1, 4, 1});

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper1) {
  auto xBuffer = new float[3]{1.f, 2.f, 3.f};
  auto xShape = new LongType[8]{2, 1, 3, 1, 1, 8192, 1, 99};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[3]{2.f, 4.f, 6.f};
  auto yShape = new LongType[8]{2, 1, 3, 1, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape);

  auto z = MmulHelper::mmul(x, y);

  ASSERT_EQ(1, z->lengthOf());
  ASSERT_NEAR(28, z->e<float>(0), 1e-5);

  delete z;
  delete[] xBuffer;
  delete[] xShape;
  delete[] yBuffer;
  delete[] yShape;
  delete y;
  delete x;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul1) {
  auto x = NDArrayFactory::create<float>('c', {6, 3});
  auto y = NDArrayFactory::create<float>('c', {3, 6});

  LongType _expS[] = {2, 3, 3, 1, 3, 8192, 1, 102};
  float _expB[] = {231.0f, 252.0f, 273.0f, 537.0f, 594.0f, 651.0f, 843.0f, 936.0f, 1029.0f};
  NDArray exp(_expB, _expS);

  for (int e = 0; e < x.lengthOf(); e++) x.p(e, e + 1);

  for (int e = 0; e < y.lengthOf(); e++) y.p(e, e + 1);

  x.permutei({1, 0});
  y.permutei({1, 0});

  auto z = MmulHelper::mmul(&x, &y);

  ASSERT_EQ(exp,*z);

  delete z;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul2) {
  auto x = NDArrayFactory::create<float>('c', {6, 3});
  auto y = NDArrayFactory::create<float>('c', {3, 6});

  LongType _expS[] = {2, 3, 3, 1, 3, 8192, 1, 102};
  float _expB[] = {231.0f, 252.0f, 273.0f, 537.0f, 594.0f, 651.0f, 843.0f, 936.0f, 1029.0f};
  NDArray exp(_expB, _expS);

  for (int e = 0; e < x.lengthOf(); e++) x.p(e, e + 1);

  for (int e = 0; e < y.lengthOf(); e++) y.p(e, e + 1);

  auto x_ = new NDArray(x.dup('f'));
  auto y_ = new NDArray(y.dup('f'));

  x_->permutei({1, 0});
  y_->permutei({1, 0});

  auto z = MmulHelper::mmul(x_, y_);

  ASSERT_EQ(exp,*z);

  delete z;
  delete x_;
  delete y_;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul3) {
  auto x = NDArrayFactory::create<float>('c', {2, 2, 2, 3, 2, 2});
  auto y = NDArrayFactory::create<float>('c', {2, 3, 2, 2});

  LongType _expS[] = {2, 8, 2, 1, 8, 8192, 1, 102};
  float _expB[] = {1624.0f, 1858.0f, 2092.0f, 2326.0f, 5368.0f,  5602.0f,  5836.0f,  6070.0f,
                   4504.0f, 5170.0f, 5836.0f, 6502.0f, 15160.0f, 15826.0f, 16492.0f, 17158.0f};
  NDArray exp(_expB, _expS);

  for (int e = 0; e < x.lengthOf(); e++) x.p(e, e + 1);

  for (int e = 0; e < y.lengthOf(); e++) y.p(e, e + 1);

  x.permutei({0, 3, 4, 5, 1, 2});
  y.permutei({3, 2, 1, 0});

  x.reshapei('c', {2 * 2 * 2, 3 * 2 * 2});
  y.reshapei('c', {2 * 2 * 3, 2});

  auto z = MmulHelper::mmul(&x, &y);

  ASSERT_EQ(exp,*z);

  delete z;
}

TEST_F(NDArrayTest, TestPermuteReshapeMmul4) {
  auto x = NDArrayFactory::create<float>('c', {2, 2, 2, 3, 2, 2});
  auto y = NDArrayFactory::create<float>('c', {2, 3, 2, 2});

  LongType _expS[] = {2, 8, 2, 1, 8, 8192, 1, 102};
  float _expB[] = {1624.0f, 1858.0f, 2092.0f, 2326.0f, 5368.0f,  5602.0f,  5836.0f,  6070.0f,
                   4504.0f, 5170.0f, 5836.0f, 6502.0f, 15160.0f, 15826.0f, 16492.0f, 17158.0f};
  NDArray exp(_expB, _expS);

  for (int e = 0; e < x.lengthOf(); e++) x.p(e, e + 1);

  for (int e = 0; e < y.lengthOf(); e++) y.p(e, e + 1);

  auto y_ = new NDArray(y.dup('f'));

  x.permutei({0, 3, 4, 5, 1, 2});
  y_->permutei({3, 2, 1, 0});

  x.reshapei('c', {2 * 2 * 2, 3 * 2 * 2});
  y_->reshapei('c', {2 * 2 * 3, 2});

  auto z = MmulHelper::mmul(&x, y_);

  ASSERT_EQ(exp,*z);

  delete z;
  delete y_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper2) {
  auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  LongType xShape[8] = {2, 5, 3, 3, 1, 8192, 1, 99};
  auto x = new NDArray(xBuffer, xShape, LaunchContext ::defaultContext(), true);

  auto yBuffer = new float[3]{2.f, 4.f, 6.f};
  LongType yShape[8] = {2, 3, 1, 1, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape, LaunchContext ::defaultContext(), true);

  auto z = NDArrayFactory::create_<float>('f', {5, 1});

  auto expBuffer = new float[5]{28.00f, 64.00f, 100.00f, 136.00f, 172.00f};
  auto exp = new NDArray(expBuffer, z->shapeInfo(), LaunchContext ::defaultContext(), true);


  MmulHelper::mmul(x, y, z);


  ASSERT_TRUE(z->equalsTo(exp));

  delete x;
  delete y;
  delete z;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper3) {
  auto xBuffer = new float[15]{1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  auto xShape = new LongType[8]{2, 5, 3, 1, 5, 8192, 1, 102};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[3]{2.f, 4.f, 6.f};
  auto yShape = new LongType[8]{2, 3, 1, 1, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape);

  auto z = NDArrayFactory::create_<float>('f', {5, 1});

  auto expBuffer = new float[5]{92.00f, 104.00f, 116.00f, 128.00f, 140.00f};
  auto exp = new NDArray(expBuffer, z->shapeInfo());



  MmulHelper::mmul(x, y, z);


  ASSERT_TRUE(z->equalsTo(exp));

  delete[] expBuffer;
  delete[] xBuffer;
  delete[] yBuffer;
  delete[] xShape;
  delete[] yShape;

  delete x;
  delete y;
  delete z;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper4) {
  auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
  auto xShape = new LongType[8]{2, 3, 2, 2, 1, 8192, 1, 99};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
  auto yShape = new LongType[8]{2, 2, 3, 3, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape);

  auto z = NDArrayFactory::create_<float>('f', {3, 3});

  auto expBuffer = new float[9]{7.0f, 21.0f, 35.0f, 10.0f, 28.0f, 46.0f, 13.0f, 35.0f, 57.0f};
  auto exp = new NDArray(expBuffer, z->shapeInfo());

  MmulHelper::mmul(x, y, z);
  ASSERT_TRUE(z->equalsTo(exp));

  delete[] expBuffer;
  delete[] xBuffer;
  delete[] yBuffer;
  delete[] xShape;
  delete[] yShape;

  delete x;
  delete y;
  delete z;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper5) {
  auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
  auto xShape = new LongType[8]{2, 3, 2, 1, 3, 8192, 1, 102};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
  auto yShape = new LongType[8]{2, 2, 3, 3, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape);

  auto z = NDArrayFactory::create_<float>('f', {3, 3});

  auto expBuffer = new float[9]{7.0f, 14.0f, 21.0f, 12.0f, 21.0f, 30.0f, 17.0f, 28.0f, 39.0f};
  auto exp = new NDArray(expBuffer, z->shapeInfo());

  MmulHelper::mmul(x, y, z);
  ASSERT_TRUE(z->equalsTo(exp));

  delete[] expBuffer;
  delete[] xBuffer;
  delete[] yBuffer;
  delete[] xShape;
  delete[] yShape;

  delete x;
  delete y;
  delete z;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper6) {
  auto xBuffer = new float[6]{1, 2, 3, 4, 5, 6};
  auto xShape = new LongType[8]{2, 3, 2, 1, 3, 8192, 1, 102};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[6]{7, 8, 9, 0, 1, 2};
  auto yShape = new LongType[8]{2, 2, 3, 1, 2, 8192, 1, 102};
  auto y = new NDArray(yBuffer, yShape);

  auto z = NDArrayFactory::create_<float>('f', {3, 3});

  auto expBuffer = new float[9]{39.0f, 54.0f, 69.0f, 9.0f, 18.0f, 27.0f, 9.0f, 12.0f, 15.0f};
  auto exp = new NDArray(expBuffer, z->shapeInfo());

  MmulHelper::mmul(x, y, z);
  ASSERT_TRUE(z->equalsTo(exp));

  delete[] expBuffer;
  delete[] xBuffer;
  delete[] yBuffer;
  delete[] xShape;
  delete[] yShape;

  delete x;
  delete y;
  delete z;
  delete exp;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMmulHelper7) {
  auto xBuffer = new float[15]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  auto xShape = new LongType[8]{2, 5, 3, 1, 5, 8192, 1, 102};
  auto x = new NDArray(xBuffer, xShape);

  auto yBuffer = new float[5]{2, 4, 6, 8, 10};
  auto yShape = new LongType[8]{2, 1, 5, 1, 1, 8192, 1, 99};
  auto y = new NDArray(yBuffer, yShape);

  auto z = NDArrayFactory::create_<float>('f', {1, 3});

  auto expBuffer = new float[9]{110.00f, 260.00f, 410.00f};
  auto exp = new NDArray(expBuffer, z->shapeInfo());

  MmulHelper::mmul(y, x, z);

  ASSERT_TRUE(z->equalsTo(exp));

  delete[] expBuffer;
  delete[] xBuffer;
  delete[] yBuffer;
  delete[] xShape;
  delete[] yShape;

  delete x;
  delete y;
  delete z;
  delete exp;
}

TEST_F(NDArrayTest, TestMmulHelper_ND_1) {
  LongType _expS[] = {3, 2, 3, 3, 9, 3, 1, 8192, 1, 99};
  float _expB[] = {70.f,   80.f,   90.f,   158.f,  184.f,  210.f,  246.f,  288.f,  330.f,
                   1030.f, 1088.f, 1146.f, 1310.f, 1384.f, 1458.f, 1590.f, 1680.f, 1770.f};

  auto a = NDArrayFactory::create<float>('c', {2, 3, 4});
  for (int e = 0; e < a.lengthOf(); e++) a.p(e, e + 1);

  auto b = NDArrayFactory::create<float>('c', {2, 4, 3});
  for (int e = 0; e < b.lengthOf(); e++) b.p(e, e + 1);

  NDArray exp(_expB, _expS);
  auto c = MmulHelper::mmul(&a, &b);

  ASSERT_TRUE(exp.isSameShape(c));
  ASSERT_TRUE(exp.equalsTo(c));

  delete c;
}

TEST_F(NDArrayTest, TestMmulHelper_ND_2) {
  LongType _expS[] = {3, 2, 72, 2, 144, 2, 1, 8192, 1, 99};
  float _expB[] = {1.07250000e+04f, 1.10500000e+04f, 2.63500000e+04f, 2.73000000e+04f, 4.19750000e+04f, 4.35500000e+04f,
                   5.76000000e+04f, 5.98000000e+04f, 7.32250000e+04f, 7.60500000e+04f, 8.88500000e+04f, 9.23000000e+04f,
                   1.04475000e+05f, 1.08550000e+05f, 1.20100000e+05f, 1.24800000e+05f, 1.35725000e+05f, 1.41050000e+05f,
                   1.51350000e+05f, 1.57300000e+05f, 1.66975000e+05f, 1.73550000e+05f, 1.82600000e+05f, 1.89800000e+05f,
                   1.98225000e+05f, 2.06050000e+05f, 2.13850000e+05f, 2.22300000e+05f, 2.29475000e+05f, 2.38550000e+05f,
                   2.45100000e+05f, 2.54800000e+05f, 2.60725000e+05f, 2.71050000e+05f, 2.76350000e+05f, 2.87300000e+05f,
                   2.91975000e+05f, 3.03550000e+05f, 3.07600000e+05f, 3.19800000e+05f, 3.23225000e+05f, 3.36050000e+05f,
                   3.38850000e+05f, 3.52300000e+05f, 3.54475000e+05f, 3.68550000e+05f, 3.70100000e+05f, 3.84800000e+05f,
                   3.85725000e+05f, 4.01050000e+05f, 4.01350000e+05f, 4.17300000e+05f, 4.16975000e+05f, 4.33550000e+05f,
                   4.32600000e+05f, 4.49800000e+05f, 4.48225000e+05f, 4.66050000e+05f, 4.63850000e+05f, 4.82300000e+05f,
                   4.79475000e+05f, 4.98550000e+05f, 4.95100000e+05f, 5.14800000e+05f, 5.10725000e+05f, 5.31050000e+05f,
                   5.26350000e+05f, 5.47300000e+05f, 5.41975000e+05f, 5.63550000e+05f, 5.57600000e+05f, 5.79800000e+05f,
                   5.73225000e+05f, 5.96050000e+05f, 5.88850000e+05f, 6.12300000e+05f, 6.04475000e+05f, 6.28550000e+05f,
                   6.20100000e+05f, 6.44800000e+05f, 6.35725000e+05f, 6.61050000e+05f, 6.51350000e+05f, 6.77300000e+05f,
                   6.66975000e+05f, 6.93550000e+05f, 6.82600000e+05f, 7.09800000e+05f, 6.98225000e+05f, 7.26050000e+05f,
                   7.13850000e+05f, 7.42300000e+05f, 7.29475000e+05f, 7.58550000e+05f, 7.45100000e+05f, 7.74800000e+05f,
                   7.60725000e+05f, 7.91050000e+05f, 7.76350000e+05f, 8.07300000e+05f, 7.91975000e+05f, 8.23550000e+05f,
                   8.07600000e+05f, 8.39800000e+05f, 8.23225000e+05f, 8.56050000e+05f, 8.38850000e+05f, 8.72300000e+05f,
                   8.54475000e+05f, 8.88550000e+05f, 8.70100000e+05f, 9.04800000e+05f, 8.85725000e+05f, 9.21050000e+05f,
                   9.01350000e+05f, 9.37300000e+05f, 9.16975000e+05f, 9.53550000e+05f, 9.32600000e+05f, 9.69800000e+05f,
                   9.48225000e+05f, 9.86050000e+05f, 9.63850000e+05f, 1.00230000e+06f, 9.79475000e+05f, 1.01855000e+06f,
                   9.95100000e+05f, 1.03480000e+06f, 1.01072500e+06f, 1.05105000e+06f, 1.02635000e+06f, 1.06730000e+06f,
                   1.04197500e+06f, 1.08355000e+06f, 1.05760000e+06f, 1.09980000e+06f, 1.07322500e+06f, 1.11605000e+06f,
                   1.08885000e+06f, 1.13230000e+06f, 1.10447500e+06f, 1.14855000e+06f, 1.12010000e+06f, 1.16480000e+06f,
                   1.13572500e+06f, 1.18105000e+06f, 1.15135000e+06f, 1.19730000e+06f, 1.16697500e+06f, 1.21355000e+06f,
                   3.54260000e+06f, 3.58980000e+06f, 3.58947500e+06f, 3.63730000e+06f, 3.63635000e+06f, 3.68480000e+06f,
                   3.68322500e+06f, 3.73230000e+06f, 3.73010000e+06f, 3.77980000e+06f, 3.77697500e+06f, 3.82730000e+06f,
                   3.82385000e+06f, 3.87480000e+06f, 3.87072500e+06f, 3.92230000e+06f, 3.91760000e+06f, 3.96980000e+06f,
                   3.96447500e+06f, 4.01730000e+06f, 4.01135000e+06f, 4.06480000e+06f, 4.05822500e+06f, 4.11230000e+06f,
                   4.10510000e+06f, 4.15980000e+06f, 4.15197500e+06f, 4.20730000e+06f, 4.19885000e+06f, 4.25480000e+06f,
                   4.24572500e+06f, 4.30230000e+06f, 4.29260000e+06f, 4.34980000e+06f, 4.33947500e+06f, 4.39730000e+06f,
                   4.38635000e+06f, 4.44480000e+06f, 4.43322500e+06f, 4.49230000e+06f, 4.48010000e+06f, 4.53980000e+06f,
                   4.52697500e+06f, 4.58730000e+06f, 4.57385000e+06f, 4.63480000e+06f, 4.62072500e+06f, 4.68230000e+06f,
                   4.66760000e+06f, 4.72980000e+06f, 4.71447500e+06f, 4.77730000e+06f, 4.76135000e+06f, 4.82480000e+06f,
                   4.80822500e+06f, 4.87230000e+06f, 4.85510000e+06f, 4.91980000e+06f, 4.90197500e+06f, 4.96730000e+06f,
                   4.94885000e+06f, 5.01480000e+06f, 4.99572500e+06f, 5.06230000e+06f, 5.04260000e+06f, 5.10980000e+06f,
                   5.08947500e+06f, 5.15730000e+06f, 5.13635000e+06f, 5.20480000e+06f, 5.18322500e+06f, 5.25230000e+06f,
                   5.23010000e+06f, 5.29980000e+06f, 5.27697500e+06f, 5.34730000e+06f, 5.32385000e+06f, 5.39480000e+06f,
                   5.37072500e+06f, 5.44230000e+06f, 5.41760000e+06f, 5.48980000e+06f, 5.46447500e+06f, 5.53730000e+06f,
                   5.51135000e+06f, 5.58480000e+06f, 5.55822500e+06f, 5.63230000e+06f, 5.60510000e+06f, 5.67980000e+06f,
                   5.65197500e+06f, 5.72730000e+06f, 5.69885000e+06f, 5.77480000e+06f, 5.74572500e+06f, 5.82230000e+06f,
                   5.79260000e+06f, 5.86980000e+06f, 5.83947500e+06f, 5.91730000e+06f, 5.88635000e+06f, 5.96480000e+06f,
                   5.93322500e+06f, 6.01230000e+06f, 5.98010000e+06f, 6.05980000e+06f, 6.02697500e+06f, 6.10730000e+06f,
                   6.07385000e+06f, 6.15480000e+06f, 6.12072500e+06f, 6.20230000e+06f, 6.16760000e+06f, 6.24980000e+06f,
                   6.21447500e+06f, 6.29730000e+06f, 6.26135000e+06f, 6.34480000e+06f, 6.30822500e+06f, 6.39230000e+06f,
                   6.35510000e+06f, 6.43980000e+06f, 6.40197500e+06f, 6.48730000e+06f, 6.44885000e+06f, 6.53480000e+06f,
                   6.49572500e+06f, 6.58230000e+06f, 6.54260000e+06f, 6.62980000e+06f, 6.58947500e+06f, 6.67730000e+06f,
                   6.63635000e+06f, 6.72480000e+06f, 6.68322500e+06f, 6.77230000e+06f, 6.73010000e+06f, 6.81980000e+06f,
                   6.77697500e+06f, 6.86730000e+06f, 6.82385000e+06f, 6.91480000e+06f, 6.87072500e+06f, 6.96230000e+06f,
                   6.91760000e+06f, 7.00980000e+06f, 6.96447500e+06f, 7.05730000e+06f, 7.01135000e+06f, 7.10480000e+06f,
                   1.17619750e+07f, 1.18560500e+07f, 1.18401000e+07f, 1.19348000e+07f, 1.19182250e+07f, 1.20135500e+07f,
                   1.19963500e+07f, 1.20923000e+07f, 1.20744750e+07f, 1.21710500e+07f, 1.21526000e+07f, 1.22498000e+07f,
                   1.22307250e+07f, 1.23285500e+07f, 1.23088500e+07f, 1.24073000e+07f, 1.23869750e+07f, 1.24860500e+07f,
                   1.24651000e+07f, 1.25648000e+07f, 1.25432250e+07f, 1.26435500e+07f, 1.26213500e+07f, 1.27223000e+07f,
                   1.26994750e+07f, 1.28010500e+07f, 1.27776000e+07f, 1.28798000e+07f, 1.28557250e+07f, 1.29585500e+07f,
                   1.29338500e+07f, 1.30373000e+07f, 1.30119750e+07f, 1.31160500e+07f, 1.30901000e+07f, 1.31948000e+07f,
                   1.31682250e+07f, 1.32735500e+07f, 1.32463500e+07f, 1.33523000e+07f, 1.33244750e+07f, 1.34310500e+07f,
                   1.34026000e+07f, 1.35098000e+07f, 1.34807250e+07f, 1.35885500e+07f, 1.35588500e+07f, 1.36673000e+07f,
                   1.36369750e+07f, 1.37460500e+07f, 1.37151000e+07f, 1.38248000e+07f, 1.37932250e+07f, 1.39035500e+07f,
                   1.38713500e+07f, 1.39823000e+07f, 1.39494750e+07f, 1.40610500e+07f, 1.40276000e+07f, 1.41398000e+07f,
                   1.41057250e+07f, 1.42185500e+07f, 1.41838500e+07f, 1.42973000e+07f, 1.42619750e+07f, 1.43760500e+07f,
                   1.43401000e+07f, 1.44548000e+07f, 1.44182250e+07f, 1.45335500e+07f, 1.44963500e+07f, 1.46123000e+07f,
                   1.45744750e+07f, 1.46910500e+07f, 1.46526000e+07f, 1.47698000e+07f, 1.47307250e+07f, 1.48485500e+07f,
                   1.48088500e+07f, 1.49273000e+07f, 1.48869750e+07f, 1.50060500e+07f, 1.49651000e+07f, 1.50848000e+07f,
                   1.50432250e+07f, 1.51635500e+07f, 1.51213500e+07f, 1.52423000e+07f, 1.51994750e+07f, 1.53210500e+07f,
                   1.52776000e+07f, 1.53998000e+07f, 1.53557250e+07f, 1.54785500e+07f, 1.54338500e+07f, 1.55573000e+07f,
                   1.55119750e+07f, 1.56360500e+07f, 1.55901000e+07f, 1.57148000e+07f, 1.56682250e+07f, 1.57935500e+07f,
                   1.57463500e+07f, 1.58723000e+07f, 1.58244750e+07f, 1.59510500e+07f, 1.59026000e+07f, 1.60298000e+07f,
                   1.59807250e+07f, 1.61085500e+07f, 1.60588500e+07f, 1.61873000e+07f, 1.61369750e+07f, 1.62660500e+07f,
                   1.62151000e+07f, 1.63448000e+07f, 1.62932250e+07f, 1.64235500e+07f, 1.63713500e+07f, 1.65023000e+07f,
                   1.64494750e+07f, 1.65810500e+07f, 1.65276000e+07f, 1.66598000e+07f, 1.66057250e+07f, 1.67385500e+07f,
                   1.66838500e+07f, 1.68173000e+07f, 1.67619750e+07f, 1.68960500e+07f, 1.68401000e+07f, 1.69748000e+07f,
                   1.69182250e+07f, 1.70535500e+07f, 1.69963500e+07f, 1.71323000e+07f, 1.70744750e+07f, 1.72110500e+07f,
                   1.71526000e+07f, 1.72898000e+07f, 1.72307250e+07f, 1.73685500e+07f, 1.73088500e+07f, 1.74473000e+07f,
                   1.73869750e+07f, 1.75260500e+07f, 1.74651000e+07f, 1.76048000e+07f, 1.75432250e+07f, 1.76835500e+07f,
                   2.46688500e+07f, 2.48098000e+07f, 2.47782250e+07f, 2.49198000e+07f, 2.48876000e+07f, 2.50298000e+07f,
                   2.49969750e+07f, 2.51398000e+07f, 2.51063500e+07f, 2.52498000e+07f, 2.52157250e+07f, 2.53598000e+07f,
                   2.53251000e+07f, 2.54698000e+07f, 2.54344750e+07f, 2.55798000e+07f, 2.55438500e+07f, 2.56898000e+07f,
                   2.56532250e+07f, 2.57998000e+07f, 2.57626000e+07f, 2.59098000e+07f, 2.58719750e+07f, 2.60198000e+07f,
                   2.59813500e+07f, 2.61298000e+07f, 2.60907250e+07f, 2.62398000e+07f, 2.62001000e+07f, 2.63498000e+07f,
                   2.63094750e+07f, 2.64598000e+07f, 2.64188500e+07f, 2.65698000e+07f, 2.65282250e+07f, 2.66798000e+07f,
                   2.66376000e+07f, 2.67898000e+07f, 2.67469750e+07f, 2.68998000e+07f, 2.68563500e+07f, 2.70098000e+07f,
                   2.69657250e+07f, 2.71198000e+07f, 2.70751000e+07f, 2.72298000e+07f, 2.71844750e+07f, 2.73398000e+07f,
                   2.72938500e+07f, 2.74498000e+07f, 2.74032250e+07f, 2.75598000e+07f, 2.75126000e+07f, 2.76698000e+07f,
                   2.76219750e+07f, 2.77798000e+07f, 2.77313500e+07f, 2.78898000e+07f, 2.78407250e+07f, 2.79998000e+07f,
                   2.79501000e+07f, 2.81098000e+07f, 2.80594750e+07f, 2.82198000e+07f, 2.81688500e+07f, 2.83298000e+07f,
                   2.82782250e+07f, 2.84398000e+07f, 2.83876000e+07f, 2.85498000e+07f, 2.84969750e+07f, 2.86598000e+07f,
                   2.86063500e+07f, 2.87698000e+07f, 2.87157250e+07f, 2.88798000e+07f, 2.88251000e+07f, 2.89898000e+07f,
                   2.89344750e+07f, 2.90998000e+07f, 2.90438500e+07f, 2.92098000e+07f, 2.91532250e+07f, 2.93198000e+07f,
                   2.92626000e+07f, 2.94298000e+07f, 2.93719750e+07f, 2.95398000e+07f, 2.94813500e+07f, 2.96498000e+07f,
                   2.95907250e+07f, 2.97598000e+07f, 2.97001000e+07f, 2.98698000e+07f, 2.98094750e+07f, 2.99798000e+07f,
                   2.99188500e+07f, 3.00898000e+07f, 3.00282250e+07f, 3.01998000e+07f, 3.01376000e+07f, 3.03098000e+07f,
                   3.02469750e+07f, 3.04198000e+07f, 3.03563500e+07f, 3.05298000e+07f, 3.04657250e+07f, 3.06398000e+07f,
                   3.05751000e+07f, 3.07498000e+07f, 3.06844750e+07f, 3.08598000e+07f, 3.07938500e+07f, 3.09698000e+07f,
                   3.09032250e+07f, 3.10798000e+07f, 3.10126000e+07f, 3.11898000e+07f, 3.11219750e+07f, 3.12998000e+07f,
                   3.12313500e+07f, 3.14098000e+07f, 3.13407250e+07f, 3.15198000e+07f, 3.14501000e+07f, 3.16298000e+07f,
                   3.15594750e+07f, 3.17398000e+07f, 3.16688500e+07f, 3.18498000e+07f, 3.17782250e+07f, 3.19598000e+07f,
                   3.18876000e+07f, 3.20698000e+07f, 3.19969750e+07f, 3.21798000e+07f, 3.21063500e+07f, 3.22898000e+07f,
                   3.22157250e+07f, 3.23998000e+07f, 3.23251000e+07f, 3.25098000e+07f, 3.24344750e+07f, 3.26198000e+07f,
                   3.25438500e+07f, 3.27298000e+07f, 3.26532250e+07f, 3.28398000e+07f, 3.27626000e+07f, 3.29498000e+07};

  auto a = NDArrayFactory::create<float>('c', {2, 72, 25});
  for (int e = 0; e < a.lengthOf(); e++) a.p(e, e + 1);

  auto b = NDArrayFactory::create<float>('c', {2, 25, 2});
  for (int e = 0; e < b.lengthOf(); e++) b.p(e, e + 1);

  NDArray exp(_expB, _expS);

  auto c = MmulHelper::mmul(&a, &b);

  ASSERT_TRUE(exp.isSameShape(c));
  ASSERT_TRUE(exp.equalsTo(c, 1e1));

  delete c;
}

TEST_F(NDArrayTest, TestNegSize1) {
  auto array = NDArrayFactory::create<float>('c', {2, 5, 7});

  ASSERT_EQ(7, array.sizeAt(-1));
  ASSERT_EQ(5, array.sizeAt(-2));
  ASSERT_EQ(2, array.sizeAt(-3));
}

//////////////////////////////////////////////////////////////////////
// not-in-place
TEST_F(NDArrayTest, Permute1) {
  LongType shape1[] = {3, 5, 10, 15, 150, 15, 1, 8192, 1, 99};
  LongType shape2[] = {3, 15, 5, 10, 1, 150, 15, 8192, 0, 99};
  const std::initializer_list<LongType> perm = {2, 0, 1};

  NDArray arr1(shape1, true);
  NDArray arr2(shape2, true);

  auto result = arr1.permute(perm);
  ASSERT_TRUE(result.isSameShapeStrict(arr2));
}

//////////////////////////////////////////////////////////////////////
// in-place
TEST_F(NDArrayTest, Permute2) {
  LongType shape1[] = {3, 5, 10, 15, 150, 15, 1, 8192, 1, 99};
  LongType shape2[] = {3, 15, 5, 10, 1, 150, 15, 8192, 0, 99};
  const std::initializer_list<LongType> perm = {2, 0, 1};

  NDArray arr1(shape1, true);
  NDArray arr2(shape2, true);

  ASSERT_TRUE(arr1.permutei(perm));
  ASSERT_TRUE(arr1.isSameShapeStrict(arr2));
}

TEST_F(NDArrayTest, RSubScalarTest1) {
  auto array = NDArrayFactory::create<double>('c', {1, 4});
  array.assign(2.0);

  auto result = NDArrayFactory::create<double>('c', {1, 4});

  array.applyScalar(scalar::ReverseSubtract, 1.0, result);

  ASSERT_NEAR(-1.0, result.meanNumber().e<double>(0), 1e-5);
}



TEST_F(NDArrayTest, TestIndexedPut2) {
  auto x = NDArrayFactory::create<float>('f', {2, 2});
  x.p(1, 1.0f);

  ASSERT_NEAR(reinterpret_cast<float *>(x.buffer())[2], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestIndexedPut3) {
  auto x = NDArrayFactory::create<float>('c', {2, 2});
  x.p(1, 1.0f);

  ASSERT_NEAR(reinterpret_cast<float *>(x.buffer())[1], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestIndexedPut4) {
  auto x = NDArrayFactory::create<float>('f', {2, 2});
  x.p(0, 1, 1.0f);

  ASSERT_NEAR(reinterpret_cast<float *>(x.buffer())[2], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestIndexedPut5) {
  auto x = NDArrayFactory::create<float>('c', {2, 2});
  x.p(0, 1, 1.0f);

  ASSERT_NEAR(x.bufferAsT<float>()[1], 1.0, 1e-5);
}

TEST_F(NDArrayTest, TestAllTensors1) {
  auto matrix = NDArrayFactory::create<float>('c', {3, 5});

  ResultSet rows = matrix.allTensorsAlongDimension({1});

  ASSERT_EQ(3, rows.size());
}

TEST_F(NDArrayTest, TestIndexing1) {
  auto matrix = NDArrayFactory::create<float>('c', {5, 5});
  for (int e = 0; e < matrix.lengthOf(); e++) matrix.p(e, (float)e);

  auto sub = matrix({2, 4, 0, 0}, true);

  ASSERT_EQ(2, sub.rows());
  ASSERT_EQ(5, sub.columns());

  ASSERT_NEAR(10, sub.e<float>(0), 1e-5);
}

TEST_F(NDArrayTest, TestIndexing2) {
  auto matrix = NDArrayFactory::create<float>('c', {2, 5, 4, 4});
  matrix.linspace(0);

  auto sub = matrix({0, 0, 2, 4, 0, 0, 0, 0}, true);

  ASSERT_EQ(2, sub.sizeAt(0));
  ASSERT_EQ(2, sub.sizeAt(1));
  ASSERT_EQ(4, sub.sizeAt(2));
  ASSERT_EQ(4, sub.sizeAt(3));

  ASSERT_EQ(64, sub.lengthOf());
  ASSERT_NEAR(32, sub.e<float>(0), 1e-5);
  ASSERT_NEAR(112, sub.e<float>(32), 1e-5);
}

TEST_F(NDArrayTest, TestIndexing3) {
  auto matrix = NDArrayFactory::create<float>('c', {5, 5});
  matrix.linspace(0);

  auto sub = matrix({2, 4, 0, 0});

  ASSERT_EQ(2, sub.rows());
  ASSERT_EQ(5, sub.columns());

  ASSERT_NEAR(10, sub.e<float>(0), 1e-5);
}

TEST_F(NDArrayTest, TestIndexing4) {
  auto matrix = NDArrayFactory::create<float>('c', {2, 5, 4, 4});
  matrix.linspace(0);

  auto sub = matrix({0, 0, 2, 4, 0, 0, 0, 0});

  ASSERT_EQ(2, sub.sizeAt(0));
  ASSERT_EQ(2, sub.sizeAt(1));
  ASSERT_EQ(4, sub.sizeAt(2));
  ASSERT_EQ(4, sub.sizeAt(3));

  ASSERT_EQ(64, sub.lengthOf());
  ASSERT_NEAR(32, sub.e<float>(0), 1e-5);
  ASSERT_NEAR(112, sub.e<float>(32), 1e-5);
}

TEST_F(NDArrayTest, TestReshapeNegative1) {
  std::unique_ptr<NDArray> array(NDArrayFactory::create_<float>('c', {2, 3, 4, 64}));

  array->reshapei('c', {-1, 64});

  ASSERT_EQ(24, array->sizeAt(0));
  ASSERT_EQ(64, array->sizeAt(1));
}

TEST_F(NDArrayTest, TestReshapeNegative2) {
  std::unique_ptr<NDArray> array(NDArrayFactory::create_<float>('c', {2, 3, 4, 64}));

  auto reshaped = array->reshape('c', {-1, 64});

  ASSERT_EQ(24, reshaped.sizeAt(0));
  ASSERT_EQ(64, reshaped.sizeAt(1));
}

//////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestStdDev1) {
  auto array = NDArrayFactory::create<double>('c', {1, 5});
  for (int e = 0; e < array.lengthOf(); e++) array.p(e, e + 1);

  auto std = array.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);
  ASSERT_NEAR(std, 1.58109, 1e-4);
}

////////////////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestStdDev2) {
  auto array = NDArrayFactory::create<double>('c', {5, 6});
  auto tad = array(0, {1});

  ASSERT_EQ(5, tad.lengthOf());

  for (int e = 0; e < tad.lengthOf(); e++) tad.p(e, e + 1);

  ASSERT_NEAR(15, tad.sumNumber().e<double>(0), 1e-5);

  auto std = tad.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);
  ASSERT_NEAR(std, 1.58109, 1e-4);
}

TEST_F(NDArrayTest, TestStdDev3) {
  auto array = NDArrayFactory::create<float>('c', {1, 50000});
  for (int e = 0; e < array.lengthOf(); e++) array.p(e, 1.f + (e % 2 ? 0.5f : -0.5f));

  auto std = array.varianceNumber(variance::SummaryStatsStandardDeviation, true).e<double>(0);
  ASSERT_NEAR(std, 0.5f, 1.0e-5f);
}

TEST_F(NDArrayTest, TestStdDev4) {
  auto array = NDArrayFactory::create<float>('c', {1, 20000});
  float const ethalon = 1 / 3.f;
  float x = ethalon;
  int total = array.lengthOf();
  for (int e = 0; e < total; e++) {
    array.p(e, 1.0f + (e % 2 ? ethalon : -ethalon));
    x *= (e % 2 ? 2.f : 0.5f);
  }
  x = 0.f;
  for (int e = 0; e < total; ++e) {
    x += array.e<float>(e);
  }
  x /= array.lengthOf();
  float y = 0;
  double M2 = 0;
  for (int e = 0; e < total; ++e) {
    M2 += (array.e<float>(e) - x) * (array.e<float>(e) - x);
  }
  M2 /= total;

  y = M2;
  auto a = array.varianceNumber(variance::SummaryStatsStandardDeviation, false);
  auto std = a.e<float>(0);
  float bY = 0.3333333f;
  ASSERT_NEAR(std, 0.3333333f, 1.0e-5f);
}

TEST_F(NDArrayTest, TestStdDev5) {
  auto array = NDArrayFactory::create<float>('c', {1, 10000});
  auto arrayD = NDArrayFactory::create<double>('c', {1, 10000});
  for (int e = 0; e < array.lengthOf(); e++) {
    array.p(e, 1.f + (e % 2 ? 1 / 5.f : -1 / 5.f));
    arrayD.p(e, 1.0 + (e % 2 ? 1 / 5. : -1 / 5.));
  }
  float stdF = array.varianceNumber(variance::SummaryStatsStandardDeviation, false).e<float>(0);
  double stdD = arrayD.varianceNumber(variance::SummaryStatsStandardDeviation, false).e<double>(0);
  ASSERT_NEAR(stdD, 0.2, 1.0e-8);    // 1/5 = 0.2
  ASSERT_NEAR(stdF, 0.2f, 1.0e-5f);  // 1/5 = 0.2
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestApplyIndexReduce1) {
  float xBuff[] = {1, 5, 2, 12, 9, 3, 10, 7, 4, 11, 6, 8};
  LongType xShapeInfo[] = {3, 2, 3, 2, 6, 2, 1, 8192, 1, 99};
  std::vector<LongType> dim = {0, 1};

  NDArray x(xBuff, xShapeInfo);
  auto exp = NDArrayFactory::create<LongType>({3, 1});

  auto result = x.applyIndexReduce(indexreduce::IndexMax, &dim);
  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyReduce3Dot) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float yBuff[] = {2, 2, 2, 2, 2, 2};
  LongType xShapeInfo[] = {2, 2, 3, 3, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, xShapeInfo);

  auto result = x.applyReduce3(reduce3::Dot, y);
  ASSERT_TRUE(result.lengthOf() == 1);
  ASSERT_NEAR(42, result.e<float>(0), 1e-5);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyAllReduce3EuclideanDistance) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float yBuff[] = {2, 2, 2, 2, 2, 2};
  float expBuff[] = {1.414214f, 1.414214f, 5.385165f, 5.385165f};
  LongType expShapeInfo[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType xShapeInfo[] = {2, 2, 3, 3, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, xShapeInfo);
  auto exp = NDArrayFactory::create<float>('c', {2, 2}, {1.414214f, 1.414214f, 5.385165f, 5.385165f});

  std::vector<LongType> dims = {1};

  auto result = x.applyAllReduce3(reduce3::EuclideanDistance, y, &dims);
  ASSERT_EQ(exp,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, applyReduce3EuclideanDistance) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float yBuff[] = {2, 2, 2, 2, 2, 2};
  float expBuff[] = {1.414214f, 1.414214f, 5.385165f, 5.385165f};
  LongType expShapeInfo[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType xShapeInfo[] = {2, 2, 3, 3, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, xShapeInfo);
  NDArray exp(expBuff, expShapeInfo);
  std::vector<LongType> dims = {1};

  auto result = x.applyAllReduce3(reduce3::EuclideanDistance, y,&dims);

  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVarianceAlongDimension1) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float expBuff[] = {0.816497f, 0.816497f};
  LongType xShapeInfo[] = {2, 2, 3, 3, 1, 8192, 1, 99};
  LongType expShapeInfo[] = {1, 2, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray exp(expBuff, expShapeInfo);
  std::vector<LongType> dims = {1};

  auto result = x.varianceAlongDimension(variance::SummaryStatsStandardDeviation, false, &dims);

  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVarianceAlongDimension2) {
  float xBuff[] = {1, 2, 3, 4, 5, 6};
  float expBuff[] = {0.666667f, 0.666667f};
  LongType xShapeInfo[] = {2, 2, 3, 3, 1, 8192, 1, 99};
  LongType expShapeInfo[] = {1, 2, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray exp(expBuff, expShapeInfo);

  std::vector<LongType> dims = {1};

  auto result = x.varianceAlongDimension(variance::SummaryStatsVariance, false, &dims);
  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}
//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVarianceAlongDimension3) {
  NDArray x = NDArrayFactory::create<double>('c', {10, 10});
  NDArray exp = NDArrayFactory::create<double>('c', {10});
  x.linspace(1);                                              // 1, 2, 3, ..., 100
  exp.assign(825.f);


  std::vector<LongType> dims = {0};

  auto result = x.varianceAlongDimension(variance::SummaryStatsVariance, false,&dims);
  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestVarianceAlongDimension4) {
  NDArray x = NDArrayFactory::create<double>('c', {12, 1, 12});  //(xBuff, xShapeInfo);
  NDArray exp = NDArrayFactory::create<double>('c', {1, 12});    //(expBuff, expShapeInfo);
  x.linspace(1);                                                 // 1, 2, 3, ..., 100
  exp.assign(1716.);
  std::vector<LongType> dims = {0};

  auto result = x.varianceAlongDimension(variance::SummaryStatsVariance, false, &dims);
  ASSERT_TRUE(exp.isSameShapeStrict(result));
  ASSERT_TRUE(exp.equalsTo(result));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestSubRowVector1) {
  float xBuff[] = {6, 7, 8, 9};
  float yBuff[] = {1, 2};
  float expBuff[] = {5, 5, 7, 7};
  LongType xShapeInfo[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType yShapeInfo[] = {2, 1, 2, 2, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, yShapeInfo);
  NDArray target(x);
  NDArray exp(expBuff, xShapeInfo);

  x.subRowVector(y, target);

  ASSERT_TRUE(exp.isSameShapeStrict(target));
  ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestDivRowVector1) {
  float xBuff[] = {6, 8, 10, 12};
  float yBuff[] = {2, 4};
  float expBuff[] = {3, 2, 5, 3};
  LongType xShapeInfo[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType yShapeInfo[] = {2, 1, 2, 2, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, yShapeInfo);
  NDArray target(x);
  NDArray exp(expBuff, xShapeInfo);

  x.divRowVector(y, target);

  ASSERT_TRUE(exp.isSameShapeStrict(target));
  ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMulRowVector1) {
  float xBuff[] = {6, 8, 10, 12};
  float yBuff[] = {2, 4};
  float expBuff[] = {12, 32, 20, 48};
  LongType xShapeInfo[] = {2, 2, 2, 2, 1, 8192, 1, 99};
  LongType yShapeInfo[] = {2, 1, 2, 2, 1, 8192, 1, 99};

  NDArray x(xBuff, xShapeInfo);
  NDArray y(yBuff, yShapeInfo);
  NDArray target(x);
  NDArray exp(expBuff, xShapeInfo);

  x.mulRowVector(y, target);

  ASSERT_TRUE(exp.isSameShapeStrict(target));
  ASSERT_TRUE(exp.equalsTo(&target));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestTensorDotAgain_1) {
  int sY = 1;
  int sX = 1;
  int pY = 0;
  int pX = 0;
  int iC = 2;
  int oC = 2;
  int kY = 3;
  int kX = 3;
  int iY = 2;
  int iX = 2;
  int oY = 6;
  int oX = 6;
  int eD = iC * oC;
  int B = 2;

  /*
  input = np.linspace(1, B * iC * iY * iX, B * iC * iY * iX).reshape(B, iC, iY, iX)
  weights = np.linspace(1, iC * oC * kY * kX, iC * oC * kY * kX).reshape(iC, oC, kY, kX)
   */
  double _expB[] = {
      96.0,  116.0, 136.0, 156.0, 256.0, 276.0, 296.0, 316.0, 102.0, 124.0, 146.0, 168.0, 278.0, 300.0, 322.0, 344.0,
      108.0, 132.0, 156.0, 180.0, 300.0, 324.0, 348.0, 372.0, 114.0, 140.0, 166.0, 192.0, 322.0, 348.0, 374.0, 400.0,
      120.0, 148.0, 176.0, 204.0, 344.0, 372.0, 400.0, 428.0, 126.0, 156.0, 186.0, 216.0, 366.0, 396.0, 426.0, 456.0,
      132.0, 164.0, 196.0, 228.0, 388.0, 420.0, 452.0, 484.0, 138.0, 172.0, 206.0, 240.0, 410.0, 444.0, 478.0, 512.0,
      144.0, 180.0, 216.0, 252.0, 432.0, 468.0, 504.0, 540.0, 150.0, 188.0, 226.0, 264.0, 454.0, 492.0, 530.0, 568.0,
      156.0, 196.0, 236.0, 276.0, 476.0, 516.0, 556.0, 596.0, 162.0, 204.0, 246.0, 288.0, 498.0, 540.0, 582.0, 624.0,
      168.0, 212.0, 256.0, 300.0, 520.0, 564.0, 608.0, 652.0, 174.0, 220.0, 266.0, 312.0, 542.0, 588.0, 634.0, 680.0,
      180.0, 228.0, 276.0, 324.0, 564.0, 612.0, 660.0, 708.0, 186.0, 236.0, 286.0, 336.0, 586.0, 636.0, 686.0, 736.0,
      192.0, 244.0, 296.0, 348.0, 608.0, 660.0, 712.0, 764.0, 198.0, 252.0, 306.0, 360.0, 630.0, 684.0, 738.0, 792.0};

  LongType _expS[] = {6, 2, 3, 3, 2, 2, 2, 72, 24, 8, 4, 2, 1, 16384, 1, 99};
  NDArray exp(_expB, _expS, LaunchContext ::defaultContext(), false);

  auto input = NDArrayFactory::create<double>('c', {B, iC, iY, iX});
  auto weights = NDArrayFactory::create<double>('c', {iC, oC, kY, kX});

  input.linspace(1);
  weights.linspace(1);

  auto result = MmulHelper::tensorDot(&weights, &input, {0}, {1});

  ASSERT_TRUE(exp.equalsTo(result));

  delete result;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestBroadcast_1) {
  double _expB[] = {1.000000, 1.000000, 1.000000, 1.000000, 2.000000, 2.000000, 2.000000, 2.000000,
                    3.000000, 3.000000, 3.000000, 3.000000, 1.000000, 1.000000, 1.000000, 1.000000,
                    2.000000, 2.000000, 2.000000, 2.000000, 3.000000, 3.000000, 3.000000, 3.000000};
  LongType _expS[] = {4, 2, 3, 2, 2, 12, 4, 2, 1, 16384, 1, 99};
  NDArray exp(_expB, _expS, LaunchContext ::defaultContext(), false);

  auto input = NDArrayFactory::create<double>('c', {2, 3, 2, 2});
  auto bias = NDArrayFactory::create<double>('c', {1, 3});

  bias.linspace(1);

  std::vector<LongType> dims = {1};

  input.applyBroadcast(broadcast::Add, &dims, bias, input);

  ASSERT_TRUE(exp.equalsTo(&input));
}

TEST_F(NDArrayTest, TestTranspose_11) {
  auto x = NDArrayFactory::create<float>('c', {2, 3, 4});
  x.transposei();

  ASSERT_EQ(4, x.sizeAt(0));
  ASSERT_EQ(3, x.sizeAt(1));
  ASSERT_EQ(2, x.sizeAt(2));
}

TEST_F(NDArrayTest, TestTranspose_12) {
  auto x = NDArrayFactory::create<float>('c', {2, 3, 4});
  auto y = x.transpose();

  ASSERT_EQ(4, y.sizeAt(0));
  ASSERT_EQ(3, y.sizeAt(1));
  ASSERT_EQ(2, y.sizeAt(2));

  ASSERT_EQ(2, x.sizeAt(0));
  ASSERT_EQ(3, x.sizeAt(1));
  ASSERT_EQ(4, x.sizeAt(2));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMMulMultiDim) {
  const int bS = 2;
  const int K = 3;
  const int N = 4;

  auto input = NDArrayFactory::create<double>('c', {bS, K, N});
  auto weights = NDArrayFactory::create<double>('c', {3 * K, K});
  auto expected = NDArrayFactory::create<double>(
      'c', {bS, 3 * K, N},
      {38,  44,  50,  56,  83,  98,   113,  128,  128,  152,  176,  200,  173,  206,  239,  272,  218,  260,
       302, 344, 263, 314, 365, 416,  308,  368,  428,  488,  353,  422,  491,  560,  398,  476,  554,  632,
       110, 116, 122, 128, 263, 278,  293,  308,  416,  440,  464,  488,  569,  602,  635,  668,  722,  764,
       806, 848, 875, 926, 977, 1028, 1028, 1088, 1148, 1208, 1181, 1250, 1319, 1388, 1334, 1412, 1490, 1568});

  input.linspace(1);
  weights.linspace(1);

  auto result = MmulHelper::mmul(&weights, &input, nullptr, 1., 0.);
  //  result must have such shape   [bS x 3K x N]

  ASSERT_TRUE(result->isSameShape(&expected));

  ASSERT_TRUE(result->equalsTo(&expected));
  delete result;
}

TEST_F(NDArrayTest, AdditionOperator1) {
  auto input1 = NDArrayFactory::create<float>('c', {2, 2});
  auto input2 = NDArrayFactory::create<float>('c', {2, 2});
  auto expected = NDArrayFactory::create<float>('c', {2, 2});

  input1.assign(1.5);
  input2.assign(2.);
  expected.assign(3.5);

  input2 = input1 + input2;

  ASSERT_TRUE(input2.equalsTo(&expected));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMatmMul_Again_1) {
  auto a = NDArrayFactory::create<float>('c', {3, 4, 1});
  auto b = NDArrayFactory::create<float>('c', {3, 1, 5});

  a.linspace(1);
  b.linspace(1);

  float _expB[] = {1.f,   2.f,   3.f,   4.f,   5.f,   2.f,   4.f,   6.f,   8.f,   10.f,  3.f,   6.f,
                   9.f,   12.f,  15.f,  4.f,   8.f,   12.f,  16.f,  20.f,  30.f,  35.f,  40.f,  45.f,
                   50.f,  36.f,  42.f,  48.f,  54.f,  60.f,  42.f,  49.f,  56.f,  63.f,  70.f,  48.f,
                   56.f,  64.f,  72.f,  80.f,  99.f,  108.f, 117.f, 126.f, 135.f, 110.f, 120.f, 130.f,
                   140.f, 150.f, 121.f, 132.f, 143.f, 154.f, 165.f, 132.f, 144.f, 156.f, 168.f, 180.f};
  LongType _expS[] = {3, 3, 4, 5, 20, 5, 1, 8192, 1, 99};
  NDArray c(_expB, _expS, LaunchContext ::defaultContext(), false);

  auto c_ = MmulHelper::mmul(&a, &b);

  ASSERT_EQ(c,*c_);

  delete c_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, TestMatmMul_Again_2) {
  auto a = NDArrayFactory::create<double>('c', {2, 5, 4});
  auto b = NDArrayFactory::create<double>('c', {2, 4, 1});

  a.linspace(1);
  b.linspace(1);

  double _expB[] = {30.f, 70.f, 110.f, 150.f, 190.f, 590.f, 694.f, 798.f, 902.f, 1006.f};
  LongType _expS[] = {3, 2, 5, 1, 5, 1, 1, 16384, 1, 99};
  NDArray c(_expB, _expS);

  auto c_ = MmulHelper::mmul(&a, &b);
  ASSERT_EQ(c,*c_);

  delete c_;
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_1) {
  double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto y = NDArrayFactory::create<double>('c', {2, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(1);
  y.linspace(1);

  auto result = x + y;
  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_2) {
  double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(1);
  y.linspace(1);

  auto result = x + y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_3) {
  double expBuff[] = {2., 3, 3., 4., 4., 5, 5., 6., 6., 7, 7., 8.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(1);
  y.linspace(1);

  auto result = x + y;
  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Plus_Test_4) {
  double expBuff[] = {11., 12., 12., 13., 13., 14., 14., 15., 13., 14., 14., 15.,
                      15., 16., 16., 17., 15., 16., 16., 17., 17., 18., 18., 19.};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto y = NDArrayFactory::create<double>('c', {4, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x + y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_1) {
  double expBuff[] = {9.,  10., 10., 11., 11., 12., 12., 13., 17., 18., 18., 19.,
                      19., 20., 20., 21., 25., 26., 26., 27., 27., 28., 28., 29.};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 2});
  auto y = NDArrayFactory::create<double>('c', {4, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_2) {
  double expBuff[] = {9., 8., 7., 6., 6.,  5.,  4.,  3.,  11., 10., 9., 8.,
                      8., 7., 6., 5., 13., 12., 11., 10., 10., 9.,  8., 7.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_3) {
  double expBuff[] = {9., 8., 7., 6., 6.,  5.,  4.,  3.,  11., 10., 9., 8.,
                      8., 7., 6., 5., 13., 12., 11., 10., 10., 9.,  8., 7.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;
  ASSERT_EQ(expected,result);

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_4) {
  double expBuff[] = {9., 10., 8., 9., 11., 12., 10., 11., 13., 14., 12., 13.};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto y = NDArrayFactory::create<double>('c', {2, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_5) {
  double expBuff[] = {9., 8, 10., 9., 11., 10, 12., 11., 13., 12, 14., 13.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;

  ASSERT_EQ(expected,result);

}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Minus_Test_6) {
  double expBuff[] = {9.,  8,  10., 9,   11., 10, 12., 11., 13., 12, 14., 13,
                      15., 14, 16., 15., 17., 16, 18., 17,  19., 18, 20., 19.};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x - y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_1) {
  double expBuff[] = {10., 11., 24., 26.,  42., 45., 64., 68., 18., 19., 40.,  42.,
                      66., 69., 96., 100., 26., 27., 56., 58., 90., 93., 128., 132.};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 2});
  auto y = NDArrayFactory::create<double>('c', {4, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_2) {
  double expBuff[] = {10., 20., 30., 40.,  55., 66., 77., 88., 12., 24., 36.,  48.,
                      65., 78., 91., 104., 14., 28., 42., 56., 75., 90., 105., 120.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_3) {
  double expBuff[] = {10., 20., 30., 40.,  55., 66., 77., 88., 12., 24., 36.,  48.,
                      65., 78., 91., 104., 14., 28., 42., 56., 75., 90., 105., 120.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_4) {
  double expBuff[] = {10., 11., 20., 22., 12., 13., 24., 26., 14., 15., 28., 30.};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto y = NDArrayFactory::create<double>('c', {2, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_5) {
  double expBuff[] = {10., 20., 11., 22., 12., 24., 13., 26., 14., 28., 15., 30.};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Multiply_Test_6) {
  double expBuff[] = {10, 11., 12., 13., 28., 30., 32., 34., 54., 57., 60., 63.};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
  auto y = NDArrayFactory::create<double>('c', {3, 1, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 1});

  x.linspace(10);
  y.linspace(1);

  auto result = x * y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_1) {
  double expBuff[] = {10.,    11.,    6., 6.5,  4.6666, 5.,  4.,  4.25, 18., 19.,     10., 10.5,
                      7.3333, 7.6666, 6., 6.25, 26.,    27., 14., 14.5, 10., 10.3333, 8.,  8.25};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 2});
  auto y = NDArrayFactory::create<double>('c', {4, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;
  ASSERT_TRUE(expected.equalsTo(result,1e-3));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_2) {
  double expBuff[] = {10., 5.,      3.333333, 2.5,   2.2, 1.83333, 1.571428, 1.375, 12., 6.,  4.,       3.,
                      2.6, 2.16666, 1.857142, 1.625, 14., 7.,      4.666666, 3.5,   3.,  2.5, 2.142857, 1.875};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_3) {
  double expBuff[] = {10., 5.,       3.333333, 2.5,   2.2, 1.833333, 1.571428, 1.375, 12., 6.,  4.,       3.,
                      2.6, 2.166666, 1.857142, 1.625, 14., 7.,       4.666666, 3.5,   3.,  2.5, 2.142857, 1.875};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {2, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_4) {
  double expBuff[] = {10., 11., 5., 5.5, 12., 13., 6., 6.5, 14., 15., 7., 7.5};

  auto x = NDArrayFactory::create<double>('c', {3, 1, 2});
  auto y = NDArrayFactory::create<double>('c', {2, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_5) {
  double expBuff[] = {10., 5., 11., 5.5, 12., 6., 13., 6.5, 14., 7., 15., 7.5};

  auto x = NDArrayFactory::create<double>('c', {3, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 2});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 2, 2});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_6) {
  double expBuff[] = {10., 5.5, 4., 3.25, 14., 7.5, 5.333333, 4.25, 18., 9.5, 6.666666, 5.25};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 4, 1});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 1});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Operator_Divide_Test_7) {
  double expBuff[] = {10., 5.,  3.333333, 2.5,  11., 5.5, 3.666666, 2.75, 12., 6.,   4., 3.,
                      13., 6.5, 4.333333, 3.25, 14., 7.,  4.666666, 3.5,  15., 7.5,  5., 3.75,
                      16., 8.,  5.333333, 4.,   17., 8.5, 5.666666, 4.25, 18., 9.,   6., 4.5,
                      19., 9.5, 6.333333, 4.75, 20., 10., 6.666666, 5.,   21., 10.5, 7., 5.25};

  auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 1, 4});
  auto expected = NDArrayFactory::create<double>(expBuff, 'c', {3, 4, 4});

  x.linspace(10);
  y.linspace(1);

  auto result = x / y;

  ASSERT_EQ(expected,result);
}

#ifndef __CUDABLAS__
//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_1) {
  auto x = NDArrayFactory::create<float>('c', {1, 5}, {1, 2, 3, 4, 5});
  auto exp = NDArrayFactory::create<float>('c', {1, 5}, {4, 5, 6, 7, 8});

  auto lambda = LAMBDA_F(_val) { return _val + 3.0f; };

  x.applyLambda<float>(lambda, x);

  ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_2) {
  auto x = NDArrayFactory::create<float>('c', {1, 5}, {1, 2, 1, 2, 1});
  auto y = NDArrayFactory::create<float>('c', {1, 5}, {1, 2, 1, 2, 1});
  auto exp = NDArrayFactory::create<float>('c', {1, 5}, {3, 5, 3, 5, 3});

  auto lambda = LAMBDA_FF(_x, _y) { return _x + _y + 1.0f; };

  x.applyPairwiseLambda<float>(y, lambda, x);

  ASSERT_TRUE(exp.equalsTo(&x));
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_Lambda_3) {
  auto x = NDArrayFactory::create<double>('c', {1, 5}, {1, 2, 1, 2, 1});
  auto y = NDArrayFactory::create<double>('c', {1, 5}, {1, 2, 1, 2, 1});
  auto exp = NDArrayFactory::create<double>('c', {1, 5}, {4, 8, 4, 8, 4});

  auto lambda = LAMBDA_DD(_x, _y) { return (_x + _y) * 2; };

  x.applyPairwiseLambda<double>(y, lambda, x);

  ASSERT_TRUE(exp.equalsTo(&x));
}

#endif

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_swapUnsafe_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 2}, {1, 2, 3, 4});
  auto y = NDArrayFactory::create<float>('c', {1, 4}, {5, 6, 7, 8});
  auto expX = NDArrayFactory::create<float>('c', {2, 2}, {5, 6, 7, 8});
  auto expY = NDArrayFactory::create<float>('c', {1, 4}, {1, 2, 3, 4});

  x.swapUnsafe(y);

  ASSERT_EQ(expX,x);
  ASSERT_EQ(expY,y);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1, 2, 3, 4, 5, 6});
  auto exp = NDArrayFactory::create<float>('c', {2, 1}, {1, 5});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_2) {
  auto x = NDArrayFactory::create<float>('f', {2, 3});
  auto exp = NDArrayFactory::create<float>('f', {2, 1}, {1, 5});
  x.linspace(1);

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_3) {
  auto x = NDArrayFactory::create<float>('c', {2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 2}, {1, 4});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_4) {
  auto x = NDArrayFactory::create<float>('f', {2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {1, 2}, {1, 4});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_5) {
  auto x = NDArrayFactory::create<float>('c', {2, 2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 2}, {1, 8});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_6) {
  auto x = NDArrayFactory::create<float>('f', {2, 2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {1, 2}, {1, 8});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_7) {
  auto x = NDArrayFactory::create<float>('f', {2, 2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {2, 1}, {1, 8});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_8) {
  auto x = NDArrayFactory::create<float>('c', {2, 3});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 2}, {1, 5});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_9) {
  auto x = NDArrayFactory::create<float>('c', {2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {2, 1}, {1, 4});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_10) {
  auto x = NDArrayFactory::create<float>('f', {2, 2});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {2, 1}, {1, 4});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_11) {
  auto x = NDArrayFactory::create<float>('f', {3, 3});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {3, 1}, {1, 5, 9});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_12) {
  auto x = NDArrayFactory::create<float>('c', {3, 3});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 3}, {1, 5, 9});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_13) {
  auto x = NDArrayFactory::create<float>('c', {3, 3, 4});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {3, 1}, {1, 18, 35});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_14) {
  auto x = NDArrayFactory::create<float>('c', {3, 3, 4});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 3}, {1, 18, 35});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_15) {
  auto x = NDArrayFactory::create<float>('f', {3, 3, 4});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {1, 3}, {1, 18, 35});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_16) {
  auto x = NDArrayFactory::create<float>('f', {1, 5});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {1, 1}, {1});

  auto diag = x.diagonal('c');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_17) {
  auto x = NDArrayFactory::create<float>('c', {5, 1});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('c', {1, 1}, {1});

  auto diag = x.diagonal('r');

  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, Test_diagonal_18) {
  auto x = NDArrayFactory::create<float>('f', {1, 1});
  x.linspace(1);
  auto exp = NDArrayFactory::create<float>('f', {1, 1}, {1});

  auto diag = x.diagonal('r');
  ASSERT_EQ(exp,diag);
}

//////////////////////////////////////////////////////////////////////
TEST_F(NDArrayTest, assign_test1) {
  NDArray x('c', {2, 3}, {1, 2, 3, 4, 5, 6});
  NDArray y('c', {2, 3}, {10, 20, 30, 40, 50, 60});
  y.reshapei('c', {3, 2});

  x.assign(y);
  x.reshapei('c', {3, 2});
  ASSERT_EQ(x,y);
}
