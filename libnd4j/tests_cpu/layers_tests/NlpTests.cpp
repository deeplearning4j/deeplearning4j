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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <ops/declarable/CustomOperations.h>
#include <NDArray.h>
#include <ops/ops.h>
#include <GradCheck.h>


using namespace nd4j;


class NlpTests : public testing::Test {
public:

    NlpTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(NlpTests, basic_sg_test_1) {
    auto target = NDArrayFactory::create<int>(0);
    auto ngStarter = NDArrayFactory::empty<int>();
    auto indices = NDArrayFactory::create<int>('c', {5}, {1, 2, 3, 4, 5});
    auto codes = NDArrayFactory::create<int8_t>('c', {5}, {1, 1, 0, 1, 1});
    auto syn0 = NDArrayFactory::create<float>('c', {100, 150});
    auto syn1 = NDArrayFactory::create<float>('c', {100, 150});
    auto syn1Neg = NDArrayFactory::empty<float>();
    auto expTable = NDArrayFactory::create<float>('c', {1000});
    auto negTable = NDArrayFactory::empty<float>();

    auto alpha = NDArrayFactory::create<double>(1.25);
    auto randomValue = NDArrayFactory::create<Nd4jLong>(119L);
    auto inferenceVector = NDArrayFactory::empty<float>();

    nd4j::ops::skipgram op;
    auto result = *op.execute({&target, &ngStarter, &indices, &codes, &syn0, &syn1, &syn1Neg, &expTable, &negTable, &alpha, &randomValue, &inferenceVector}, {}, {}, {false});
    ASSERT_EQ(Status::OK(), result.status());
}