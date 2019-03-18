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


class DeclarableOpsTests13 : public testing::Test {
public:

    DeclarableOpsTests13() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests13, test_pow_1) {
    auto x = NDArrayFactory::create<float>('c', {2, 2}, {2.f, 2.f, 2.f, 2.f});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 3});
    auto e = NDArrayFactory::create<float>('c', {2, 2}, {8.f, 8.f, 8.f, 8.f});

    nd4j::ops::Pow op;
    auto result = op.execute({&x, &y}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);

    ASSERT_EQ(e, *z);

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_1) {
    auto start = NDArrayFactory::create<int>(0);
    auto limit = NDArrayFactory::create<int>(0);

    nd4j::ops::range op;
    auto result = op.execute({&start, &limit}, {}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_2) {

    nd4j::ops::range op;
    auto result = op.execute({}, {1.0, 1.0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_empty_range_3) {

    nd4j::ops::range op;
    auto result = op.execute({}, {}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    auto z = result->at(0);
    ASSERT_TRUE(z->isEmpty());

    delete result;
}

TEST_F(DeclarableOpsTests13, test_argmax_edge_1) {
    auto ctx = new Context(1);
    auto arr = NDArrayFactory::create_<float>('c', {1024,1});

    ctx->setInputArray(0, arr, true);
    ctx->setOutputArray(0, NDArrayFactory::create_<Nd4jLong >('c', {1}), true);
    ctx->setInputArray(1, NDArrayFactory::create_<Nd4jLong >(0), true);   //Axis 0


    nd4j::ops::argmax op;
    auto result = op.execute(ctx);

    nd4j_printf("Done\n","");
    delete ctx;
}

TEST_F(DeclarableOpsTests13, test_listdiff_1) {
    auto x = NDArrayFactory::create<int>('c', {4}, {0, 1, 2, 3});
    auto y = NDArrayFactory::create<int>('c', {2}, {3, 1});

    auto od = NDArrayFactory::create<int>('c', {2});
    auto oi = NDArrayFactory::create<int>('c', {2});

    nd4j::ops::listdiff op;
    auto result = op.execute({&x, &y}, {&od, &oi}, {}, {}, {});
    ASSERT_EQ(Status::OK(), result);
}

TEST_F(DeclarableOpsTests13, test_or_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, true, true, true});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::Or, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_and_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, false, false, true});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::And, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests13, test_xor_1) {
    auto x = NDArrayFactory::create<bool>('c', {4}, {false, true, false, true});
    auto y = NDArrayFactory::create<bool>('c', {4}, {false, false, true, true});
    auto e = NDArrayFactory::create<bool>('c', {4}, {false, true, true, false});

    auto z = NDArrayFactory::create<bool>('c', {4});

    x.applyPairwiseTransform(pairwise::Xor, &y, &z, nullptr);

    ASSERT_EQ(e, z);
}