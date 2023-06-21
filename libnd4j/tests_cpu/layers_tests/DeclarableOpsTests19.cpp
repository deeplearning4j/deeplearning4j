/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
//
#include <ops/declarable/CustomOperations.h>
#include <ops/ops.h>
#include <indexing/NDIndexUtils.h>

#include "testlayers.h"

using namespace sd;

class DeclarableOpsTests19 : public testing::Test {
 public:
  DeclarableOpsTests19() {
    printf("\n");
    fflush(stdout);
  }
};

TEST_F(DeclarableOpsTests19, test_argmax_maxint_vector_1) {
  auto x = NDArrayFactory::create<float>('c', {3}, {0.1f, 0.5f, 0.7f});
  auto z = NDArrayFactory::create<sd::LongType>(0);
  auto e = NDArrayFactory::create<sd::LongType>(2);

  sd::ops::argmax op;
  auto status = op.execute({&x}, {&z}, {DataTypeUtils::max<int>()});
  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_EQ(e, z);
}




TEST_F(DeclarableOpsTests19, test_matmul_ccc) {
  auto x = NDArrayFactory::create<float>('c', {10, 10});
  auto y = NDArrayFactory::create<float>('c', {10, 10});
  auto e = NDArrayFactory::create<float>('c', {10, 10});
  auto z = NDArrayFactory::create<float>('c', {10, 10});

  z.assign(100.f);
  e.assign(110.f);
  x.assign(1.0f);
  y.assign(1.0f);

  sd::ops::matmul op;
  auto status = op.execute({&x, &y}, {&z}, {1.0, 1.0});
  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests19, test_matmul_fcf) {
  auto x = NDArrayFactory::create<float>('f', {10, 10});
  auto y = NDArrayFactory::create<float>('c', {10, 10});
  auto e = NDArrayFactory::create<float>('f', {10, 10});
  auto z = NDArrayFactory::create<float>('f', {10, 10});

  z.assign(100.f);
  e.assign(110.f);
  x.assign(1.0f);
  y.assign(1.0f);

  sd::ops::matmul op;
  auto status = op.execute({&x, &y}, {&z}, {1.0, 1.0});
  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests19, test_matmul_cff) {
  auto x = NDArrayFactory::create<float>('c', {10, 10});
  auto y = NDArrayFactory::create<float>('f', {10, 10});
  auto e = NDArrayFactory::create<float>('f', {10, 10});
  auto z = NDArrayFactory::create<float>('f', {10, 10});

  z.assign(100.f);
  e.assign(110.f);
  x.assign(1.0f);
  y.assign(1.0f);

  sd::ops::matmul op;
  auto status = op.execute({&x, &y}, {&z}, {1.0, 1.0});
  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests19, test_matmul_ccf) {
  auto x = NDArrayFactory::create<float>('c', {10, 10});
  auto y = NDArrayFactory::create<float>('c', {10, 10});
  auto e = NDArrayFactory::create<float>('f', {10, 10});
  auto z = NDArrayFactory::create<float>('f', {10, 10});

  z.assign(100.f);
  e.assign(110.f);
  x.assign(1.0f);
  y.assign(1.0f);

  sd::ops::matmul op;
  auto status = op.execute({&x, &y}, {&z}, {1.0, 1.0});
  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests19, test_matmul_fff) {
  auto x = NDArrayFactory::create<float>('f', {10, 10});
  auto y = NDArrayFactory::create<float>('f', {10, 10});
  auto e = NDArrayFactory::create<float>('f', {10, 10});
  auto z = NDArrayFactory::create<float>('f', {10, 10});

  z.assign(100.f);
  e.assign(110.f);
  x.assign(1.0f);
  y.assign(1.0f);

  sd::ops::matmul op;
  auto status = op.execute({&x, &y}, {&z}, {1.0, 1.0});
  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests19, test_conv1d_bp_1) {
  /*
  DynamicCustomOp op = DynamicCustomOp.builder("conv1d_bp")
          .addInputs(
                  Nd4j.create(DataType.FLOAT, 2,2,12),
                  Nd4j.create(DataType.FLOAT, 3,2,3),
                  Nd4j.create(DataType.FLOAT, 2,3,6)
          )
          .addOutputs(
                  Nd4j.create(DataType.FLOAT, 2,2,12),
                  Nd4j.create(DataType.FLOAT, 3,2,3))
          .addIntegerArguments(3,2,0,1,2,0)
          .build();

  Nd4j.exec(op);
   */

  auto t = NDArrayFactory::create<float>('c', {2, 2, 12});
  auto u = NDArrayFactory::create<float>('c', {3, 2, 3});
  auto v = NDArrayFactory::create<float>('c', {2, 3, 6});

  sd::ops::conv1d_bp op;
  auto result = op.evaluate({&t, &u, &v}, {3, 2, 0, 1, 2, 0});
  ASSERT_EQ(sd::Status::OK, result.status());
}

TEST_F(DeclarableOpsTests19, test_squeeze_1) {
  auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
  auto e = NDArrayFactory::create<double>('c', {3, 4});
  int axis = 2;

  sd::ops::squeeze op;
  auto status = op.execute({&x}, {&e}, {axis});
  ASSERT_EQ(sd::Status::OK, status);
}


TEST_F(DeclarableOpsTests19, test_create_view_1) {
  auto xLinspace = NDArrayFactory::linspace<double>(1,12,12);
  auto x = xLinspace->reshape('c',{3,4});
  //multiple parts:
  //index type: 0 = point,interval = 1,all = 2,new axis = 3
  auto indexFirstPoint = sd::NDIndexUtils::createPoint(1);

  sd::ops::create_view op;
  auto result = op.evaluate({&x,&indexFirstPoint,&indexFirstPoint});
  result.setNonRemovable();
  auto shape = result[0]->getShapeAsVectorInt();
  auto expectedShape = std::vector<int>({});
  ASSERT_EQ(expectedShape,shape);

  auto assertion = NDArrayFactory::create<double>(6.0);
  ASSERT_TRUE(assertion.equalsTo(result[0],1e-5));


}


TEST_F(DeclarableOpsTests19,test_create_view_2) {
   sd::ops::create_view op;
   auto inclusive = std::vector<int>({0,1});

  for(int i = 0; i < 2; i++) {
    auto x = NDArrayFactory::create<double>('c', {3, 4});
    auto all = sd::NDIndexUtils::createAll();
    auto indexInterval = sd::NDIndexUtils::createInterval(0,1,1,(sd::LongType ) inclusive[i]);
    auto expectedRows = inclusive[i] > 0 ? 2 : 1;
    auto expectedShapeInterval = std::vector<int>({expectedRows,4});
    auto resultInterval = op.evaluate({&x,&indexInterval,&all});
    auto shapeInterval = resultInterval[0]->getShapeAsVectorInt();
    ASSERT_EQ(expectedShapeInterval,shapeInterval);
    resultInterval.setNonRemovable();
  }


}

TEST_F(DeclarableOpsTests19,test_create_view_3) {
  sd::ops::create_view op;
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto expectedShapeAll = std::vector<int>({3,4});
  auto all = sd::NDIndexUtils::createAll();
  auto newAll = all.dup();
  auto resultAll = op.evaluate({&x,&all,&newAll});
  resultAll.setNonRemovable();
  auto shapeAll = resultAll[0]->getShapeAsVectorInt();
  ASSERT_EQ(expectedShapeAll,shapeAll);


}


TEST_F(DeclarableOpsTests19,test_create_view_4) {
  sd::ops::create_view op;
  auto expectedShapeAll2 = std::vector<int>({3,4});
  auto x = NDArrayFactory::create<double>('c', {3, 4});
  auto all = sd::NDIndexUtils::createAll();

  auto newAll2 = all.dup();
  auto resultAll2 = op.evaluate({&x,&all});
  resultAll2.setNonRemovable();
  auto shapeAll2 = resultAll2[0]->getShapeAsVectorInt();
  ASSERT_EQ(expectedShapeAll2,shapeAll2);

}

TEST_F(DeclarableOpsTests19,test_create_view_5) {
  sd::ops::create_view op;
  auto vectorInput = NDArrayFactory::create<double>(1.0);
  auto newAxis = sd::NDIndexUtils::createNewAxis();
  auto resultNewAxis = op.evaluate({&vectorInput,&newAxis});
  auto expectedNewAxis = NDArrayFactory::create<double>(1.0);
  auto newExpectedAxis = expectedNewAxis.reshape('c',{1});
  ASSERT_EQ(newExpectedAxis.getShapeAsVectorInt(),resultNewAxis[0]->getShapeAsVectorInt());
  resultNewAxis.setNonRemovable();
}

TEST_F(DeclarableOpsTests19,test_create_view_6) {
  sd::ops::create_view op;
  auto linspace = NDArrayFactory::linspace<double>(1,125,125);
  auto reshaped = linspace->reshape('c',{5,5,5});
  auto slice = sd::NDIndexUtils::createInterval(0,1,1,false);
  auto resultSlice = op.evaluate({&reshaped,&slice});
  resultSlice.setNonRemovable();
  auto assertionShape = std::vector<int>({1,5,5});
  auto resultSliceShape = resultSlice[0]->getShapeAsVectorInt();
  ASSERT_EQ(assertionShape,resultSliceShape);

}

TEST_F(DeclarableOpsTests19,test_create_view_7) {
  sd::ops::create_view op;
  //intervals, new axis, point, all
  auto fiveByFive = NDArrayFactory::linspace<double>(1,25,25);
  auto reshapedFiveByFive = fiveByFive->reshape('c',{5,5});
  auto columns = NDIndexUtils::createInterval(0,1,1,false);
  auto newAll4 = NDIndexUtils::createAll();
  auto columnVectorAssertion = std::vector<int>({5,1});
  auto resultFiveByFive = op.evaluate({&reshapedFiveByFive,&newAll4,&columns});
  resultFiveByFive.setNonRemovable();
  auto resultFiveByFiveShape = resultFiveByFive[0]->getShapeAsVectorInt();
  ASSERT_EQ(columnVectorAssertion,resultFiveByFiveShape);

}

TEST_F(DeclarableOpsTests19,test_create_view_8) {
  sd::ops::create_view op;
  auto fiveByFiveSubColumns = NDArrayFactory::linspace<double>(1,25,25);
  auto reshapedFiveByFiveSubColumns = fiveByFiveSubColumns->reshape('c',{5,5});
  auto columns2 = NDIndexUtils::createInterval(0,1,1,false);
  auto newAll3 = NDIndexUtils::createPoint(1);
  auto subColumnsAssertion = std::vector<int>({1});
  auto resultSubColumn = op.evaluate({&reshapedFiveByFiveSubColumns,&columns2,&newAll3});
  resultSubColumn.setNonRemovable();
  auto subColumnShape = resultSubColumn[0]->getShapeAsVectorInt();
  ASSERT_EQ(subColumnsAssertion,subColumnShape);
}


TEST_F(DeclarableOpsTests19,test_create_view_9) {
  sd::ops::create_view op;
  auto fiveByFiveSubColumns = NDArrayFactory::linspace<double>(1,25,25);
  auto reshapedFiveByFiveSubColumns = fiveByFiveSubColumns->reshape('c',{5,5});
  auto columns2 = NDIndexUtils::createInterval(0,1,1,false);
  auto newAll3 = NDIndexUtils::createPoint(1);
  auto subColumnsAssertion = std::vector<int>({1});
  auto resultSubColumn = op.evaluate({&reshapedFiveByFiveSubColumns,&columns2,&newAll3});
  resultSubColumn.setNonRemovable();
  auto subColumnShape = resultSubColumn[0]->getShapeAsVectorInt();
  ASSERT_EQ(subColumnsAssertion,subColumnShape);
}