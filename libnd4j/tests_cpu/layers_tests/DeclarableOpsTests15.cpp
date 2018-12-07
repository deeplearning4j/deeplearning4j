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


class DeclarableOpsTests15 : public testing::Test {
public:

    DeclarableOpsTests15() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(DeclarableOpsTests15, Test_Half_assign_1) {
    auto x = NDArrayFactory::create<float16>('c', {2, 5});
    int y = 1;
    x.assign(y);

    ASSERT_EQ(10, x.sumNumber().e<int>(0));
}