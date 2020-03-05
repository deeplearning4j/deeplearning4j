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


class DeclarableOpsTests19 : public testing::Test {
public:

    DeclarableOpsTests19() {
        printf("\n");
        fflush(stdout);
    }
};

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
    ASSERT_EQ(Status::OK(), result->status());


    delete result;
}

TEST_F(DeclarableOpsTests19, test_squeeze_1) {
    auto x = NDArrayFactory::create<double>('c', {3, 4, 1});
    auto e = NDArrayFactory::create<double>('c', {3, 4});
    int axis = 2;

    sd::ops::squeeze op;
    auto status = op.execute({&x}, {&e}, {axis});
    ASSERT_EQ(Status::OK(), status);
}