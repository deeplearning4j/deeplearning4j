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


using namespace sd;


class DeclarableOpsTests18 : public testing::Test {
public:

    DeclarableOpsTests18() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests18, test_bitcast_1) {
    auto x = NDArrayFactory::create<double>(0.23028551377579154);
    auto z = NDArrayFactory::create<Nd4jLong>(0);
    auto e = NDArrayFactory::create<Nd4jLong>(4597464930322771456L);

    sd::ops::bitcast op;
    auto status = op.execute({&x}, {&z}, {}, {(Nd4jLong) sd::DataType::INT64}, {});
    ASSERT_EQ(Status::OK(), status);

    ASSERT_EQ(e, z);
}

TEST_F(DeclarableOpsTests18, test_tanh_1) {
    auto x = NDArrayFactory::create<float>('c', {8}, {0.23f, -0.23f, 0.35f, -0.35f, 0.64f, -0.64f, 100000.f, -100000.f});
    auto z = x.ulike();
    auto e = NDArrayFactory::create<float>('c', {8}, {0.226028f, -0.226028f, 0.336376f, -0.336376f, 0.564900f, -0.564900f, 1.f, -1.f});

    sd::ops::tanh op;
    op.execute({&x}, {&z});

    ASSERT_EQ(e, z);
}