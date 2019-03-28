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
// Created by raver on 8/4/2018.
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>


using namespace nd4j;


class DeclarableOpsTests14 : public testing::Test {
public:

    DeclarableOpsTests14() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests14, Test_Validation_Edge_1) {
    auto x = NDArrayFactory::create<int>('c', {2}, {2, 2});
    auto exp = NDArrayFactory::create('c', {2, 2}, Environment::getInstance()->defaultFloatDataType());
    exp.assign(4.0f);

    nd4j::ops::fill op;
    auto result = op.execute({&x}, {4.0f},{}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(exp, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Reshape_CF_1) {
    auto x = NDArrayFactory::create<double>('f', {2, 3}, {1.0, 4.0, 2.0, 5.0, 3.0, 6.0});
    auto e = NDArrayFactory::create<double>('f', {3, 2}, {1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

    x.printShapeInfo("x shape");
    x.printBuffer("x buffr");
    x.printIndexedBuffer("x indxd");

    auto r = x.reshape('c', {3, 2});
    r->printIndexedBuffer("r pre-s");
    r->streamline('f');    

    nd4j::ops::reshape op;
    auto result = op.execute({&x}, {}, {3, 2}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    z->printShapeInfo("z shape");
    z->printBuffer("z buffr");
    z->printIndexedBuffer("z indxd");
    printf("------------------\n");
    r->printShapeInfo("r shape");
    r->printBuffer("r buffr");
    r->printIndexedBuffer("r indxd");

    delete r;
    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_1) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});

    ASSERT_EQ(x, y);
}

TEST_F(DeclarableOpsTests14, Test_Inf_Comparison_2) {
    auto x = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, 1.0/0.0, 5});
    auto y = NDArrayFactory::create<double>('c', {5}, {1, 2, 3, -1.0/0.0, 5});

    ASSERT_NE(x, y);
}

TEST_F(DeclarableOpsTests14, Test_EvalReductionShape_1) {
    auto x = NDArrayFactory::create<int>('c', {3}, {5, 3, 4});
    auto y = NDArrayFactory::create<int>('c', {1}, {1});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {2}, {5, 4});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.execute({&x, &y}, {}, {}, {false, false});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_EvalReductionShape_2) {
    auto x = NDArrayFactory::create<int>('c', {3}, {5, 3, 4});
    auto y = NDArrayFactory::create<int>('c', {1}, {1});
    auto e = NDArrayFactory::create<Nd4jLong>('c', {3}, {5, 1, 4});

    nd4j::ops::evaluate_reduction_shape op;
    auto result = op.execute({&x, &y}, {}, {}, {true, false});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_Diag_Zeros_1) {
    auto x = NDArrayFactory::create<double>('c', {2}, {1, 2});
    auto z = NDArrayFactory::create<double>('c', {2, 2}, {-119, -119, -119, -119});
    auto exp = NDArrayFactory::create<double>('c', {2, 2}, {1, 0, 0, 2});

    nd4j::ops::diag op;
    auto status = op.execute({&x}, {&z}, {}, {}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(exp, z);
}

TEST_F(DeclarableOpsTests14, Test_scalar_broadcast_1) {
    auto x = NDArrayFactory::create<float>(1.0f);
    auto y = NDArrayFactory::create<float>('c', {5, 10});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    e.assign(1.0);


    nd4j::ops::add op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}

TEST_F(DeclarableOpsTests14, Test_scalar_broadcast_2) {
    auto x = NDArrayFactory::create<float>(1.0f);
    auto y = NDArrayFactory::create<float>('c', {5, 10});
    auto e = NDArrayFactory::create<float>('c', {5, 10});
    y.assign(2.0f);
    e.assign(-1.0f);


    nd4j::ops::subtract op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    ASSERT_EQ(e, *result->at(0));

    delete result;
}