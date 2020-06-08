/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
#include <ops/declarable/CustomOperations.h>
#include <array/NDArray.h>
#include <ops/ops.h>
#include <helpers/GradCheck.h>
#include <array>
#include <helpers/RandomLauncher.h>


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
    auto z = NDArrayFactory::create<Nd4jLong>(0);
    auto e = NDArrayFactory::create<Nd4jLong>(2);

    sd::ops::argmax op;
    auto status = op.execute({&x}, {&z}, {DataTypeUtils::max<int>()});
    ASSERT_EQ(Status::OK(), status);
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

    //encoded->printIndexedBuffer("ENC");

    ASSERT_EQ(exp_encoded, *encoded);
    ASSERT_EQ(exp_gradients, x);

    // FIXME: we need to add a way to declare individual inplace outputs
    //ASSERT_EQ(exp_gradients, *gradients);
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
    ASSERT_EQ(Status::OK(), status);
    ASSERT_EQ(exp_gradients, x);
}

TEST_F(DeclarableOpsTests19, test_bitmap_encode_1) {
    auto initial = NDArrayFactory::create<float>('c', {6}, {0.0f, 0.0f, 1e-3f, -1e-3f, 0.0f, 0.0f});
    auto exp_0 = initial.like();
    auto exp_1 = initial.dup();
    auto exp_c = NDArrayFactory::create<int>(2L);

    sd::ops::encode_bitmap enc;
    auto enc_result = enc.evaluate({&initial}, {1e-3f});
    ASSERT_EQ(Status::OK(), enc_result.status());

    //initial.printIndexedBuffer("initial");
    ASSERT_EQ(exp_0, initial);

    auto encoded = enc_result.at(1);
    auto counter = enc_result.at(2);

    //encoded->printIndexedBuffer("encoded");

    ASSERT_EQ(exp_c, *counter);

    sd::ops::decode_bitmap dec;
    auto status = dec.execute({&initial, encoded}, {&initial});
    ASSERT_EQ(Status::OK(), status);


    //initial.printIndexedBuffer();

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
            nd4j_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
    }

    ASSERT_NE(exp, initial);
    ASSERT_EQ(neg, initial);

    sd::ops::decode_bitmap dec;
    auto status = dec.execute({&initial, encoded}, {&initial});
    ASSERT_EQ(Status::OK(), status);

    // checking equality of all dedoded bits
    for (int e = 0; e < initial.lengthOf(); e++) {
        auto f = initial.e<float>(e);
        if (f != 1.0f)
            nd4j_printf("initial[%i] = %f\n", e, f);
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
            nd4j_printf("initial[%i] = %f\n", e, f);
            throw std::runtime_error("");
        }
    }
    ASSERT_EQ(neg, initial);

    // checking equality of all encoded bits
    //for (int e = 5; e < encoded->lengthOf() - 1; e++) {
        //if (encoded->e<int>(e) != encoded->e<int>(e - 1) + 1)
            //nd4j_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
    //}

    sd::ops::decode_threshold dec;
    auto status = dec.execute({&initial, encoded}, {&initial});
    ASSERT_EQ(Status::OK(), status);

    // checking equality of all dedoded bits
    for (int e = 0; e < initial.lengthOf(); e++) {
        auto f = initial.e<float>(e);
        if (f != 1.0f)
            nd4j_printf("initial[%i] = %f\n", e, f);
    }

    ASSERT_EQ(exp, initial);
}

#ifdef _RELEASE
TEST_F(DeclarableOpsTests19, test_threshold_encode_decode_2) {
  // [2,1,135079944,1,1,8192,1,99]
  auto initial = NDArrayFactory::create<float>('c', {1, 135079944});
  initial = 1.0f;
  auto exp = initial.dup();
  auto neg = initial.like();
  neg = 0.5f;

  sd::ops::encode_threshold enc;
  auto enc_result = enc.evaluate({&initial}, {0.5f});
  auto encoded = enc_result.at(1);

  ASSERT_EQ(135079944 + 4, encoded->lengthOf());
  ASSERT_NE(exp, initial);
/*
  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 0.5f) {
      nd4j_printf("initial[%i] = %f\n", e, f);
      throw std::runtime_error("");
    }
  }
  */
  ASSERT_EQ(neg, initial);

  // checking equality of all encoded bits
  //for (int e = 5; e < encoded->lengthOf() - 1; e++) {
  //if (encoded->e<int>(e) != encoded->e<int>(e - 1) + 1)
  //nd4j_printf("Non equal encoded values at E[%i]: %i;\n", e, encoded->e<int>(e));
  //}

  sd::ops::decode_threshold dec;
  auto status = dec.execute({&initial, encoded}, {&initial});
  ASSERT_EQ(Status::OK(), status);

  // checking equality of all dedoded bits
  /*
  for (int e = 0; e < initial.lengthOf(); e++) {
    auto f = initial.e<float>(e);
    if (f != 1.0f)
      nd4j_printf("initial[%i] = %f\n", e, f);
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
    ASSERT_EQ(Status::OK(), status);

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
    ASSERT_EQ(Status::OK(), status);

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
    ASSERT_EQ(Status::OK(), status);

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
    ASSERT_EQ(Status::OK(), status);

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
    ASSERT_EQ(Status::OK(), status);

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
    auto result = op.evaluate({&t, &u, &v}, {3, 2, 0, 1, 2,0});
    ASSERT_EQ(Status::OK(), result.status());

}

TEST_F(DeclarableOpsTests19, test_squeeze_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
    auto e = NDArrayFactory::create<double>('c', {3, 4});
    int axis = 2;

    sd::ops::squeeze op;
    auto status = op.execute({&x}, {&e}, {axis});
    ASSERT_EQ(Status::OK(), status);
}
