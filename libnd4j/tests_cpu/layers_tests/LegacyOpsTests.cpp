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
// Created by raver119 on 16.10.2017.
//
#include <array/NDArray.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/ShapeUtils.h>

#include <loops/reduce3.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyTransformOp.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;

class LegacyOpsTests : public NDArrayTests {};

TEST_F(LegacyOpsTests, TransformTests_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);
  auto z = NDArrayFactory::create<float>('c', {5, 5});
  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(-1.0);

  LegacyTransformSameOp op(transform::Neg);  // Neg
  auto status = op.execute({&x}, {&z}, {}, {}, {});
  ASSERT_EQ(status, sd::Status::OK);
  ASSERT_TRUE(z.equalsTo(&exp));
}

TEST_F(LegacyOpsTests, TransformTests_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);

  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(-1.0);

  LegacyTransformSameOp op(transform::Neg);  // Neg
  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);

  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(LegacyOpsTests, Reciprocal_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(2.0f);

  auto ethalon = NDArrayFactory::create<float>('c', {5, 5});
  ethalon.assign(0.5f);

  LegacyTransformSameOp op(transform::Reciprocal);  // Reciprocal
  Status status = op.execute({&x}, {&x}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_TRUE(ethalon.equalsTo(&x));
}

TEST_F(LegacyOpsTests, PWT_Tests_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(2.0);

  auto y = NDArrayFactory::create<float>('c', {5, 5});
  y.assign(3.0);

  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(6.0);

  LegacyPairwiseTransformOp op(pairwise::Multiply);  // Multiply
  Status status = op.execute({&x, &y}, {&x}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, PWT_Tests_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(2.0);

  auto y = NDArrayFactory::create<float>('c', {5, 5});
  y.assign(3.0);

  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(6.0);

  LegacyPairwiseTransformOp op(pairwise::Multiply);  // Multiply
  auto result = op.evaluate({&x, &y}, {}, {});

  auto z = result.at(0);

  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(LegacyOpsTests, Scalar_Test_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(2.0);

  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(7.0);

  LegacyScalarOp op(scalar::Add);
  op.execute({&x}, {&x}, {5.0}, {}, {});  //

  ASSERT_TRUE(exp.equalsTo(&x));
}

TEST_F(LegacyOpsTests, Scalar_Test_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(2.0);

  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  exp.assign(7.0);

  auto y = NDArrayFactory::create<float>(5.0f);

  LegacyScalarOp op(scalar::Add, y);
  auto result = op.evaluate({&x}, {}, {});

  auto z = result.at(0);
  ASSERT_TRUE(exp.equalsTo(z));
}

TEST_F(LegacyOpsTests, ReduceTests_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);
  int opNum = reduce::Sum;
  LegacyReduceSameOp op(opNum);

  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);
  ASSERT_TRUE(z->isScalar());
  ASSERT_NEAR(x.sumNumber().e<float>(0), z->e<float>(0), 1e-5f);
}

TEST_F(LegacyOpsTests, ReduceTests_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);

  LegacyReduceSameOp op(reduce::Sum);
  auto axis = NDArrayFactory::create<LongType>('c', {1}, {1});
  auto result = op.evaluate({&x, &axis}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);

  std::vector<LongType> dims = {1};
  auto exp = x.reduceAlongDimension(reduce::Sum, &dims);

ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, ReduceTests_3) {
  auto x = NDArrayFactory::create<float>('c', {3, 5});
  x.linspace(1);
  auto indices = NDArrayFactory::create<int>('c', {1, 1}, {1});

  LegacyReduceSameOp op(reduce::Sum);
  auto result = op.evaluate({&x, &indices}, {}, {});
  auto z = result.at(0);
  std::vector<LongType> dims = {1};

  auto exp = x.reduceAlongDimension(reduce::Sum, &dims);

  ASSERT_EQ(sd::Status::OK, result.status());

ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, ReduceTests_4) {
  auto x = NDArrayFactory::create<float>('c', {2, 3, 5});
  x.linspace(1);
  auto indices = NDArrayFactory::create<int>('c', {1, 1}, {1});

  LegacyReduceSameOp op(reduce::Sum);
  auto result = op.evaluate({&x, &indices}, {}, {}, {true});
  auto z = result.at(0);
  std::vector<LongType> dims = {1};
  auto exp = x.reduceAlongDimension(reduce::Sum,&dims, true);
  ASSERT_EQ(sd::Status::OK, result.status());
ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, ReduceTests_5) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);
  int opNum = reduce::Mean;
  LegacyReduceFloatOp op(opNum);

  auto result = op.evaluate({&x});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);
  ASSERT_TRUE(z->isScalar());
  ASSERT_NEAR(x.meanNumber().e<float>(0), z->e<float>(0), 1e-5f);
}

TEST_F(LegacyOpsTests, ReduceTests_6) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(1.0);
  auto axis = NDArrayFactory::create<int>('c', {1}, {1});
  LegacyReduceFloatOp op(reduce::Mean);

  auto result = op.evaluate({&x, &axis}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);
  std::vector<LongType> dims = {1};

  auto exp = x.reduceAlongDimension(reduce::Mean,&dims);

ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, ReduceTests_7) {
  auto x = NDArrayFactory::create<float>('c', {3, 5});
  x.linspace(1);
  auto indices = NDArrayFactory::create<int>('c', {1, 1}, {1});

  LegacyReduceFloatOp op(reduce::Mean);
  auto result = op.evaluate({&x, &indices}, {}, {});
  auto z = result.at(0);
  std::vector<LongType> dims = {1};
  auto exp = x.reduceAlongDimension(reduce::Mean, &dims);

  ASSERT_EQ(sd::Status::OK, result.status());

ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, ReduceTests_8) {
  auto x = NDArrayFactory::create<float>('c', {2, 3, 5});
  x.linspace(1);
  auto indices = NDArrayFactory::create<int>('c', {1}, {1});

  LegacyReduceFloatOp op(reduce::Mean);
  auto result = op.evaluate({&x, &indices}, {}, {}, {true});
  auto z = result.at(0);
  std::vector<LongType> dims = {1};
  auto exp = x.reduceAlongDimension(reduce::Mean, &dims, true);

  ASSERT_EQ(sd::Status::OK, result.status());
ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, IndexReduceTests_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  x.linspace(1);

  LegacyIndexReduceOp op(indexreduce::IndexMax);

  auto result = op.evaluate({&x}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);

  ASSERT_TRUE(z->isScalar());
  ASSERT_EQ(24, z->e<int>(0));
}

TEST_F(LegacyOpsTests, IndexReduceTests_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  auto indices = NDArrayFactory::create<int>('c', {1}, {1});
  x.linspace(1);
  std::vector<LongType> shape = {5,1};
  auto exp = NDArrayFactory::create<LongType>({4, 4, 4, 4, 4});
  exp.reshapei(shape);
  LegacyIndexReduceOp op(indexreduce::IndexMax);

  auto result = op.evaluate({&x, &indices}, {}, {});

  ASSERT_EQ(1, result.size());

  auto z = result.at(0);
  ASSERT_EQ(exp,*z);
}

TEST_F(LegacyOpsTests, BroadcastingTests_1) {
  auto x = NDArrayFactory::create<double>('c', {5, 5});
  x.assign(0.0f);

  auto row = NDArrayFactory::create<double>('c', { 5});
  row.linspace(1);
  auto axis = NDArrayFactory::create<LongType>('c', {1}, {1});
  LegacyBroadcastOp op(broadcast::Add);
  Status status = op.execute({&x, &row, &axis}, {&x}, {}, {}, {});

  ASSERT_EQ(sd::Status::OK, status);

  auto list = x.allTensorsAlongDimension({1});
  for (int e = 0; e < list.size(); e++) ASSERT_TRUE(row.equalsTo(list.at(e)));
}

TEST_F(LegacyOpsTests, BroadcastingTests_2) {
  auto x = NDArrayFactory::create<double>('c', {5}, {1, 1, 1, 1, 1});
  auto y = NDArrayFactory::create<double>('c', {10, 5});
  auto e = NDArrayFactory::create<double>('c', {10, 5});
  y.assign(3.0);
  e.assign(4.0);

  LongType axis = 1;

  auto packY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), {axis});

  NDArray::prepareSpecialUse({&y}, {&x});

  NativeOpExecutioner::execInverseBroadcast(
      LaunchContext::defaultContext(), broadcast::Add, x.buffer(), x.shapeInfo(), x.specialBuffer(),
      x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), y.buffer(),
      y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), &axis, 1, packY->platformShapeInfo(),
      packY->platformOffsets(), packY->platformShapeInfo(), packY->platformOffsets());

  NDArray::registerSpecialUse({&y}, {&x});

  ASSERT_EQ(e, y);
}

TEST_F(LegacyOpsTests, PowDerivative_1) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  auto exp = NDArrayFactory::create<float>('c', {5, 5});
  x.assign(3.f);
  exp.assign(6.f);

  float p = 2.0f;

  x.applyScalar(scalar::PowDerivative, p, x);

  ASSERT_TRUE(exp.equalsTo(&x));
}

#ifndef SD_CUDA
TEST_F(LegacyOpsTests, reduce3_1) {
  sd::LongType yShape[2] = {4, 4};
  sd::LongType xShape[1] = {4};
  float y[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float x[4] = {1, 2, 3, 4};
  sd::LongType dimension[1] = {1};
  sd::LongType dimensionLength = 1;
  int opNum = 1;
  float extraVals[1] = {0};
  float result[4] = {0.0, 0.0, 0.0, 0.0};

  std::vector<sd::LongType> dim = {1};

  auto shapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 2, yShape);
  auto xShapeBuffer = sd::ShapeBuilders::createShapeInfo(sd::DataType::FLOAT32, 'c', 1, xShape);

  auto tadShapeBuffer = sd::ShapeUtils::evalReduceShapeInfo('c', &dim, shapeBuffer, false, true, nullptr);
  functions::reduce3::Reduce3<float, float>::exec(opNum, x, xShapeBuffer, extraVals, y, shapeBuffer, result,
                                                  tadShapeBuffer, dimension, dimensionLength, 0, 4);

  float distancesAssertion[4] = {0.0, 8.0, 16.0, 24.0};
  for (int i = 0; i < 4; i++) ASSERT_NEAR(distancesAssertion[i], result[i], 1e-5);

  delete[] shapeBuffer;
  delete[] xShapeBuffer;
}

#endif

TEST_F(LegacyOpsTests, Reduce3_2) {
  auto x = NDArrayFactory::create<float>('c', {5, 5});
  auto y = NDArrayFactory::create<float>('c', {5});
  auto z = NDArrayFactory::create<float>('c', {5});

  auto dim = NDArrayFactory::create<int>('c', {1}, {1});
  dim.syncToHost();

  LaunchContext* context = LaunchContext::defaultContext();

  Pointer* extraPointers = nullptr;
#ifdef SD_CUDA
  extraPointers = new Pointer[7]{nullptr,
                                     context->getCudaStream(),
                                     context->getScalarPointer(),
                                     nullptr,
                                     context->getCudaSpecialStream(),
                                     context->getReductionPointer(),
                                     context->getAllocationPointer()};
#endif

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), {1});
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), {1});

  NDArray::prepareSpecialUse({&z}, {&x, &y, &dim});
  OpaqueDataBuffer xBuf(x.dataBuffer());
  OpaqueDataBuffer yBuf(y.dataBuffer());
  OpaqueDataBuffer zBuf(z.dataBuffer());
  OpaqueDataBuffer dimBuf(dim.dataBuffer());

  execReduce3Tad(extraPointers, reduce3::CosineSimilarity, &xBuf, x.shapeInfo(), x.specialShapeInfo(), nullptr, &yBuf,
                 y.shapeInfo(), y.specialShapeInfo(), &zBuf, z.shapeInfo(), z.specialShapeInfo(), &dimBuf,
                 dim.shapeInfo(), dim.specialShapeInfo(), packX->platformShapeInfo(), packX->platformOffsets(),
                 packY->platformShapeInfo(), packY->platformOffsets());

  NDArray::registerSpecialUse({&z}, {&x, &y, &dim});

  delete[] extraPointers;
}



TEST_F(LegacyOpsTests, Reduce3_4) {
  auto x =
      NDArrayFactory::create<double>('c', {3, 5},
                                     {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                                      -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                                      0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247});

  auto y = NDArrayFactory::create<double>(
      'c', {1, 5}, {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673});
  auto e = NDArrayFactory::create<double>('c', {1, 3}, {0.577452, 0.0, 1.80182});
  auto z = NDArrayFactory::create<double>('c', {1, 3});

  auto dim = NDArrayFactory::create<int>('c', {1}, {1});
  dim.syncToHost();

  LaunchContext* context = LaunchContext::defaultContext();

  Pointer* extraPointers = nullptr;
#ifdef SD_CUDA
  extraPointers = new Pointer[7]{nullptr,
                                     context->getCudaStream(),
                                     context->getScalarPointer(),
                                     nullptr,
                                     context->getCudaSpecialStream(),
                                     context->getReductionPointer(),
                                     context->getAllocationPointer()};
#endif

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), {1});
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), {1});

  NDArray::prepareSpecialUse({&z}, {&x, &y, &dim});
  OpaqueDataBuffer xBuf(x.dataBuffer());
  OpaqueDataBuffer yBuf(y.dataBuffer());
  OpaqueDataBuffer zBuf(z.dataBuffer());
  OpaqueDataBuffer dimBuf(dim.dataBuffer());

  execReduce3Tad(extraPointers, reduce3::CosineDistance, &xBuf, x.shapeInfo(), x.specialShapeInfo(), nullptr, &yBuf,
                 y.shapeInfo(), y.specialShapeInfo(), &zBuf, z.shapeInfo(), z.specialShapeInfo(), &dimBuf,
                 dim.shapeInfo(), dim.specialShapeInfo(), packX->platformShapeInfo(), packX->platformOffsets(),
                 packY->platformShapeInfo(), packY->platformOffsets());

  NDArray::registerSpecialUse({&z}, {&x, &y, &dim});
  ASSERT_EQ(e, z);
  delete[] extraPointers;
}

TEST_F(LegacyOpsTests, Reduce3_5) {
  auto x =
      NDArrayFactory::create<double>('c', {3, 5},
                                     {-0.84443557262, -0.06822254508, 0.74266910552, 0.61765557527, -0.77555125951,
                                      -0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673,
                                      0.62955373525, -0.31357592344, 1.03362500667, -0.59279078245, 1.1914824247});

  auto y = NDArrayFactory::create<double>(
      'c', {1, 5}, {-0.99536740779, -0.0257304441183, -0.6512106060, -0.345789492130, -1.25485503673});
  auto e = NDArrayFactory::create<double>('c', {1, 3}, {0.577452, 0.0, 1.80182});
  auto z = NDArrayFactory::create<double>('c', {1, 3});

  auto dim = NDArrayFactory::create<int>('c', {1}, {1});
  dim.syncToHost();

  LaunchContext* context = LaunchContext::defaultContext();

  Pointer* extraPointers = nullptr;
#ifdef SD_CUDA
  extraPointers = new Pointer[7]{nullptr,
                                     context->getCudaStream(),
                                     context->getScalarPointer(),
                                     nullptr,
                                     context->getCudaSpecialStream(),
                                     context->getReductionPointer(),
                                     context->getAllocationPointer()};
#endif

  auto packX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), {1});
  auto packY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), {1});

  NDArray::prepareSpecialUse({&z}, {&x, &y, &dim});

  OpaqueDataBuffer xBuf(x.dataBuffer());
  OpaqueDataBuffer yBuf(y.dataBuffer());
  OpaqueDataBuffer zBuf(z.dataBuffer());
  OpaqueDataBuffer dimBuf(dim.dataBuffer());

  execReduce3Tad(extraPointers, reduce3::CosineDistance, &xBuf, x.shapeInfo(), x.specialShapeInfo(), nullptr, &yBuf,
                 y.shapeInfo(), y.specialShapeInfo(), &zBuf, z.shapeInfo(), z.specialShapeInfo(), &dimBuf,
                 dim.shapeInfo(), dim.specialShapeInfo(), packX->platformShapeInfo(), packX->platformOffsets(),
                 packY->platformShapeInfo(), packY->platformOffsets());

  NDArray::registerSpecialUse({&z}, {&x, &y, &dim});
  ASSERT_EQ(e, z);
  delete[] extraPointers;
}

TEST_F(LegacyOpsTests, test_Reduce3_All_1) {
  auto x = NDArrayFactory::create<float>('c', {1000, 100});
  auto y = NDArrayFactory::create<float>('c', {1, 100});
  auto z = NDArrayFactory::create<float>('c', {1000, 1});
  auto dim = NDArrayFactory::create<int>('c', {1}, {-1});

  auto tadPackX = ConstantTadHelper::getInstance().tadForDimensions(x.shapeInfo(), -1);
  auto tadPackY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), -1);

  LaunchContext* context = LaunchContext::defaultContext();

  Pointer* extraPointers = nullptr;
#ifdef SD_CUDA
  extraPointers = new Pointer[7]{nullptr,
                                     context->getCudaStream(),
                                     context->getScalarPointer(),
                                     nullptr,
                                     context->getCudaSpecialStream(),
                                     context->getReductionPointer(),
                                     context->getAllocationPointer()};
#endif

  NDArray::prepareSpecialUse({&z}, {&x, &y});

  OpaqueDataBuffer xBuf(x.dataBuffer());
  OpaqueDataBuffer yBuf(y.dataBuffer());
  OpaqueDataBuffer zBuf(z.dataBuffer());
  OpaqueDataBuffer dimBuf(dim.dataBuffer());

  execReduce3All(extraPointers, reduce3::EuclideanDistance, &xBuf, x.shapeInfo(), x.specialShapeInfo(), nullptr, &yBuf,
                 y.shapeInfo(), y.specialShapeInfo(), &zBuf, z.shapeInfo(), z.specialShapeInfo(), &dimBuf,
                 dim.shapeInfo(), dim.specialShapeInfo(), tadPackX->platformShapeInfo(), tadPackX->platformOffsets(),
                 tadPackY->platformShapeInfo(), tadPackY->platformOffsets());

  NDArray::registerSpecialUse({&z}, {&x, &y});

  delete[] extraPointers;
}

TEST_F(LegacyOpsTests, test_inverse_broadcast_1) {
  auto x = NDArrayFactory::create<float>('c', {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  auto y = NDArrayFactory::create<float>('c', {3, 4});
  auto e = NDArrayFactory::create<float>('c', {3, 4});
  e.assign(2.0f);

  auto tadPackY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), 1);

  y.tickWriteDevice();

  NativeOpExecutioner::execInverseBroadcast(
      LaunchContext::defaultContext(), broadcast::Add, x.buffer(), x.shapeInfo(), x.specialBuffer(),
      x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), y.buffer(),
      y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), nullptr, 0, tadPackY->platformShapeInfo(),
      tadPackY->platformOffsets(), tadPackY->platformShapeInfo(), tadPackY->platformOffsets());

  ASSERT_EQ(e, y);
}

TEST_F(LegacyOpsTests, test_inverse_broadcast_2) {
  auto x = NDArrayFactory::create<float>('c', {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  auto y = NDArrayFactory::create<float>('c', {3, 4});
  auto z = NDArrayFactory::create<bool>('c', {3, 4});
  auto e = NDArrayFactory::create<bool>('c', {3, 4});
  e.assign(false);

  auto row = y(1, {0});
  row.assign(2.0f);

  auto erow = e(1, {0});
  erow.assign(true);

  auto tadPackY = ConstantTadHelper::getInstance().tadForDimensions(y.shapeInfo(), 1);

  z.tickWriteDevice();

  NativeOpExecutioner::execInverseBroadcastBool(
      LaunchContext::defaultContext(), broadcast::BoolOps::EqualTo, x.buffer(), x.shapeInfo(), x.specialBuffer(),
      x.specialShapeInfo(), y.buffer(), y.shapeInfo(), y.specialBuffer(), y.specialShapeInfo(), z.buffer(),
      z.shapeInfo(), z.specialBuffer(), z.specialShapeInfo(), nullptr, nullptr, 0, tadPackY->platformShapeInfo(),
      tadPackY->platformOffsets(), tadPackY->platformShapeInfo(), tadPackY->platformOffsets());

  ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, test_legacy_reduce_empty_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 0, 3});
  auto z = NDArrayFactory::create<float>('c', {2, 3});
  auto e = NDArrayFactory::create<float>('c', {2, 3});

  LongType dim = 1;

  NativeOpExecutioner::execReduceSame(LaunchContext::defaultContext(), reduce::SameOps::Sum, x.buffer(), x.shapeInfo(),
                                      x.specialBuffer(), x.specialShapeInfo(), nullptr, z.buffer(), z.shapeInfo(),
                                      z.specialBuffer(), z.specialShapeInfo(), &dim, 1);

  ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, test_legacy_reduce_empty_2) {
  auto x = NDArrayFactory::create<float>('c', {2, 0, 3});
  auto z = NDArrayFactory::create<float>('c', {2, 3});
  auto e = NDArrayFactory::create<float>('c', {2, 3});
  e.assign(std::numeric_limits<float>::infinity());

  LongType dim = 1;

  NativeOpExecutioner::execReduceSame(LaunchContext::defaultContext(), reduce::SameOps::Min, x.buffer(), x.shapeInfo(),
                                      x.specialBuffer(), x.specialShapeInfo(), nullptr, z.buffer(), z.shapeInfo(),
                                      z.specialBuffer(), z.specialShapeInfo(), &dim, 1);

  ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, test_legacy_reduce_empty_3) {
  auto x = NDArrayFactory::create<float>('c', {2, 0, 3});
  auto z = NDArrayFactory::create<float>('c', {2, 3});
  auto e = NDArrayFactory::create<float>('c', {2, 3});
  e.assign(-std::numeric_limits<float>::infinity());

  LongType dim = 1;

  NativeOpExecutioner::execReduceSame(LaunchContext::defaultContext(), reduce::SameOps::Max, x.buffer(), x.shapeInfo(),
                                      x.specialBuffer(), x.specialShapeInfo(), nullptr, z.buffer(), z.shapeInfo(),
                                      z.specialBuffer(), z.specialShapeInfo(), &dim, 1);

  ASSERT_EQ(e, z);
}

TEST_F(LegacyOpsTests, test_legacy_reduce_empty_4) {
  if (!Environment::getInstance().isCPU()) return;
  int a = 0;

  auto x = NDArrayFactory::create<float>('c', {1, 0, 2});
  auto d = NDArrayFactory::create<int>('c', {1}, {a});
  auto z = NDArrayFactory::create<float>('c', {0, 2});
  auto e = NDArrayFactory::create<float>('c', {0, 2});

  InteropDataBuffer xdb(x.dataBuffer());
  InteropDataBuffer ddb(d.dataBuffer());
  InteropDataBuffer zdb(z.dataBuffer());

  execReduceSame2(nullptr, reduce::SameOps::Sum, &xdb, x.shapeInfo(), x.specialShapeInfo(), nullptr, &zdb,
                    z.shapeInfo(), z.specialShapeInfo(), &ddb, d.shapeInfo(), d.specialShapeInfo());
}

TEST_F(LegacyOpsTests, test_legacy_transform_float_1) {
  auto x = NDArrayFactory::create<float>('c', {1, 0, 4});

  NativeOpExecutioner::execTransformFloat(LaunchContext::defaultContext(), transform::FloatOps::RSqrt, x.buffer(),
                                          x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), x.buffer(),
                                          x.shapeInfo(), x.specialBuffer(), x.specialShapeInfo(), nullptr, nullptr,
                                          nullptr);
}
