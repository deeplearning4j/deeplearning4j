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
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>
#include <helpers/RandomLauncher.h>


using namespace nd4j;


class AttentionTests : public testing::Test {
public:
    AttentionTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(AttentionTests, basic_dot_product_attention) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 1});

    nd4j::ops::dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values}, {1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

/*
//Ignored: AB 2019/05/21 - Segmentation fault on on linux-ppc64le-cpu - https://github.com/deeplearning4j/deeplearning4j/issues/7657
TEST_F(AttentionTests, basic_dot_product_attention_bp) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 1});
    auto eps = NDArrayFactory::create<float>('c', {10, 4, 1});

    nd4j::ops::dot_product_attention_bp op;
    auto result = op.execute({&queries, &keys, &values, &eps}, {}, {1, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
*/

TEST_F(AttentionTests, basic_dot_product_attention_with_weights) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 1});

    nd4j::ops::dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values}, {1, 1});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

TEST_F(AttentionTests, basic_dot_product_attention_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 1});
    auto mask = NDArrayFactory::create<float>('c', {10, 3});
    mask.assign(1.);

    nd4j::ops::dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values, &mask}, {1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

/*
//AB 2019/05/28 - Segfault on ppc64le - See issue #7657
TEST_F(AttentionTests, basic_dot_product_attention_bp_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 1});
    auto eps = NDArrayFactory::create<float>('c', {10, 4, 1});
    auto mask = NDArrayFactory::create<float>('c', {10, 3});
    mask.assign(1.);

    nd4j::ops::dot_product_attention_bp op;
    auto result = op.execute({&queries, &keys, &values, &eps, &mask}, {}, {1, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
 */

TEST_F(AttentionTests, multi_head_input_dot_product_attention_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {2, 5, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {2, 5, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {2, 5, 4, 1});
    auto mask = NDArrayFactory::create<float>('c', {2, 3});
    mask.assign(1.);

    nd4j::ops::dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values, &mask}, {1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

/*
//AB 2019/05/30 - Segfault on ppc64le - See issue #7657
TEST_F(AttentionTests, multi_head_input_dot_product_attention_bp_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {2, 5, 4, 3});
    auto values = NDArrayFactory::create<float>('c', {2, 5, 4, 3});
    auto queries = NDArrayFactory::create<float>('c', {2, 5, 4, 1});
    auto eps = NDArrayFactory::create<float>('c', {2, 5, 4, 1});
    auto mask = NDArrayFactory::create<float>('c', {2, 3});
    mask.assign(1.);

    nd4j::ops::dot_product_attention_bp op;
    auto result = op.execute({&queries, &keys, &values, &eps, &mask}, {}, {1, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
 */


TEST_F(AttentionTests, basic_multi_head_dot_product_attention) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 2});

    auto Wk = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wv = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wq = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wo = NDArrayFactory::create<float>('c', {2* 3, 4});

    nd4j::ops::multi_head_dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values, &Wk, &Wv, &Wq, &Wo}, {1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

/*
//AB 2019/05/30 - Other attention BP tests are segfaulting on ppc64le - disabling this pre-emptively - See issue #7657
TEST_F(AttentionTests, basic_multi_head_dot_product_bp_attention) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 2});

    auto Wk = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wv = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wq = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wo = NDArrayFactory::create<float>('c', {2* 3, 7});

    auto eps = NDArrayFactory::create<float>('c', {10, 7, 2});


    nd4j::ops::multi_head_dot_product_attention_bp op;
    auto result = op.execute({&queries, &keys, &values, &Wk, &Wv, &Wq, &Wo, &eps}, {}, {1, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
 */

TEST_F(AttentionTests, basic_multi_head_dot_product_attention_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 2});

    auto Wk = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wv = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wq = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wo = NDArrayFactory::create<float>('c', {2* 3, 4});

    auto mask = NDArrayFactory::create<float>('c', {10, 5});
    mask.assign(1.);


    nd4j::ops::multi_head_dot_product_attention op;
    auto result = op.evaluate({&queries, &keys, &values, &Wk, &Wv, &Wq, &Wo, &mask}, {1, 0});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}

/*
//AB 2019/05/30 - Other attention BP tests are segfaulting on ppc64le - disabling this pre-emptively - See issue #7657
TEST_F(AttentionTests, basic_multi_head_dot_product_bp_attention_with_mask) {
    auto keys = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto values = NDArrayFactory::create<float>('c', {10, 4, 5});
    auto queries = NDArrayFactory::create<float>('c', {10, 4, 2});

    auto Wk = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wv = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wq = NDArrayFactory::create<float>('c', {2, 3, 4});
    auto Wo = NDArrayFactory::create<float>('c', {2* 3, 7});

    auto eps = NDArrayFactory::create<float>('c', {10, 7, 2});

    auto mask = NDArrayFactory::create<float>('c', {10, 5});
    mask.assign(1.);


    nd4j::ops::multi_head_dot_product_attention_bp op;
    auto result = op.execute({&queries, &keys, &values, &Wk, &Wv, &Wq, &Wo, &eps, &mask}, {}, {1, 0}, {});
    ASSERT_EQ(Status::OK(), result->status());

    delete result;
}
 */
