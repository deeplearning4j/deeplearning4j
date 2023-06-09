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

TEST_F(DeclarableOpsTests19, test_threshold_encode_1) {
  auto x = NDArrayFactory::create<double>('c', {3}, {1.5, 2.5, -3.5});
  auto exp_encoded = NDArrayFactory::create<int>('c', {7}, {3, 3, 1056964608, 0, 1, 2, -3});
  auto exp_gradients = NDArrayFactory::create<double>('c', {3}, {1.0, 2.0, -3.0});

  sd::ops::encode_threshold op;
  auto result = op.evaluate({&x}, {0.5});

  auto gradients = result.at(0);
  auto encoded = result.at(1);

  // encoded->printIndexedBuffer("ENC");

  ASSERT_EQ(exp_encoded, *encoded);
  ASSERT_EQ(exp_gradients, x);

  // FIXME: we need to add a way to declare individual inplace outputs
  // ASSERT_EQ(exp_gradients, *gradients);
}

TEST_F(DeclarableOpsTests19, test_threshold_encode_2) {
  for (int length = 5; length < 35; length++) {
    auto x = NDArrayFactory::create<double>('c', {10000});
    auto exp_gradients = NDArrayFactory::create<double>('c', {10000});

    for (int e = 0; e < length; e++) {
      x.p(e, 2e-3);
      exp_gradients.p(e, 1e-3);
    }

    sd::ops::encode_threshold op;
    auto result = op.evaluate({&x}, {1e-3});

    auto encoded = result.at(1);

    ASSERT_EQ(length + 4, encoded->lengthOf());
    ASSERT_EQ(exp_gradients, x);
  }
}

TEST_F(DeclarableOpsTests19, test_threshold_encode_boundary_1) {
  auto x = NDArrayFactory::create<float>('c', {6});
  x = 1.0f;

  sd::ops::encode_threshold op;
  auto result = op.evaluate({&x}, {1.0}, {3});

  auto gradients = result.at(0);
  auto encoded = result.at(1);

  ASSERT_EQ(7, encoded->lengthOf());
  ASSERT_EQ(3, x.sumNumber().e<int>(0));
}

TEST_F(DeclarableOpsTests19, test_threshold_encode_boundary_2) {
  auto x = NDArrayFactory::create<float>('c', {1000});
  x = 1.0f;

  sd::ops::encode_threshold op;
  auto result = op.evaluate({&x}, {1.0}, {100});

  auto gradients = result.at(0);
  auto encoded = result.at(1);

  ASSERT_EQ(104, encoded->lengthOf());

  ASSERT_EQ(900, x.sumNumber().e<int>(0));
}

TEST_F(DeclarableOpsTests19, test_threshold_decode_1) {
  auto x = NDArrayFactory::create<double>('c', {3}, {1.0, 2.0, -3.0});
  auto y = NDArrayFactory::create<int>('c', {7}, {3, 3, 1056964608, 0, 1, 2, -3});
  auto exp_gradients = NDArrayFactory::create<double>('c', {3}, {1.5, 2.5, -3.5});

  sd::ops::decode_threshold op;
  auto status = op.execute({&x, &y}, {&x});
  ASSERT_EQ(sd::Status::OK, status);
  ASSERT_EQ(exp_gradients, x);
}

TEST_F(DeclarableOpsTests19, test_bitmap_encode_1) {
  auto initial = NDArrayFactory::create<float>('c', {6}, {0.0f, 0.0f, 1e-3f, -1e-3f, 0.0f, 0.0f});
  auto exp_0 = initial.like();
  auto exp_1 = initial.dup();
  auto exp_c = NDArrayFactory::create<int>(2L);

  sd::ops::encode_bitmap enc;
  auto enc_result = enc.evaluate({&initial}, {1e-3f});
  ASSERT_EQ(sd::Status::OK, enc_result.status());

  // initial.printIndexedBuffer("initial");
  ASSERT_EQ(exp_0, initial);

  auto encoded = enc_result.at(1);
  auto counter = enc_result.at(2);

  // encoded->printIndexedBuffer("encoded");

  ASSERT_EQ(exp_c, *counter);

  sd::ops::decode_bitmap dec;
  auto status = dec.execute({&initial, encoded}, {&initial});
  ASSERT_EQ(sd::Status::OK, status);

  // initial.printIndexedBuffer();

  ASSERT_EQ(exp_1, initial);
}

TEST_F(DeclarableOpsTests19, test_bitmap_encode_decode) {
  auto initial = NDArrayFactory::create<float>('c', {256000});
  initial = 1.0f;
  auto exp = initial.dup();
  auto neg = initial.like();
  neg = 0.5f;

  sd::ops::encode_bitmap enc;
  auto enc_result = enc.evaluate({&initial}, {0.5f});
  auto encoded = enc_result.at(1);

  // checking equality of all encoded bits
  for (int e = 5; e < encoded->lengthOf() - 1; e++) {
    if (encoded->e<int>(e) != encoded->e<int>(e - 1))
      sd_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
  }

  ASSERT_NE(exp, initial);
  ASSERT_EQ(neg, initial);

  sd::ops::decode_bitmap dec;
  auto status = dec.execute({&initial, encoded}, {&initial});
  ASSERT_EQ(sd::Status::OK, status);

  // checking equality of all dedoded bits
  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 1.0f) sd_printf("initial[%i] = %f\n", e, f);
  }

  ASSERT_EQ(exp, initial);
}

TEST_F(DeclarableOpsTests19, test_threshold_encode_decode) {
  auto initial = NDArrayFactory::create<float>('c', {256000});
  initial = 1.0f;
  auto exp = initial.dup();
  auto neg = initial.like();
  neg = 0.5f;

  sd::ops::encode_threshold enc;
  auto enc_result = enc.evaluate({&initial}, {0.5f});
  auto encoded = enc_result.at(1);

  ASSERT_EQ(256000 + 4, encoded->lengthOf());
  ASSERT_NE(exp, initial);

  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 0.5f) {
      sd_printf("initial[%i] = %f\n", e, f);
      THROW_EXCEPTION("");
    }
  }
  ASSERT_EQ(neg, initial);

  // checking equality of all encoded bits
  // for (int e = 5; e < encoded->lengthOf() - 1; e++) {
  // if (encoded->e<int>(e) != encoded->e<int>(e - 1) + 1)
  // sd_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
  //}

  sd::ops::decode_threshold dec;
  auto status = dec.execute({&initial, encoded}, {&initial});
  ASSERT_EQ(sd::Status::OK, status);

  // checking equality of all dedoded bits
  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 1.0f) sd_printf("initial[%i] = %f\n", e, f);
  }

  ASSERT_EQ(exp, initial);
}

#ifdef _RELEASE
TEST_F(DeclarableOpsTests19, test_threshold_encode_decode_2) {
  // [2,1,135079944,1,1,8192,1,99]
  constexpr int sizeX = 10 * 1000 * 1000;
  auto initial = NDArrayFactory::create<float>('c', {1, sizeX});
  initial = 1.0f;
  auto exp = initial.dup();
  auto neg = initial.like();
  neg = 0.5f;

  sd::ops::encode_threshold enc;
  auto enc_result = enc.evaluate({&initial}, {0.5f});
  auto encoded = enc_result.at(1);

  ASSERT_EQ(sizeX + 4, encoded->lengthOf());
  ASSERT_NE(exp, initial);
  /*
    for (int e = 0; e < initial.lengthOf(); e++) {
      auto f = initial.e<float>(e);
      if (f != 0.5f) {
        sd_printf("initial[%i] = %f\n", e, f);
        THROW_EXCEPTION("");
      }
    }
    */
  ASSERT_EQ(neg, initial);

  // checking equality of all encoded bits
  // for (int e = 5; e < encoded->lengthOf() - 1; e++) {
  // if (encoded->e<int>(e) != encoded->e<int>(e - 1) + 1)
  // sd_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
  //}

  sd::ops::decode_threshold dec;
  auto status = dec.execute({&initial, encoded}, {&initial});
  ASSERT_EQ(sd::Status::OK, status);

  // checking equality of all dedoded bits
  /*
  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 1.0f)
      sd_printf("initial[%i] = %f\n", e, f);
  }
   */

  ASSERT_EQ(exp, initial);
}
#endif

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
    auto indexInterval = sd::NDIndexUtils::createInterval(0,1,1,inclusive[i]);
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
  auto slice = sd::NDIndexUtils::createInterval(0,1);
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
  auto columns = NDIndexUtils::createInterval(0,1,1,0);
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
  auto columns2 = NDIndexUtils::createInterval(0,1,1,0);
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
  auto columns2 = NDIndexUtils::createInterval(0,1,1,0);
  auto newAll3 = NDIndexUtils::createPoint(1);
  auto subColumnsAssertion = std::vector<int>({1});
  auto resultSubColumn = op.evaluate({&reshapedFiveByFiveSubColumns,&columns2,&newAll3});
  resultSubColumn.setNonRemovable();
  auto subColumnShape = resultSubColumn[0]->getShapeAsVectorInt();
  ASSERT_EQ(subColumnsAssertion,subColumnShape);
}