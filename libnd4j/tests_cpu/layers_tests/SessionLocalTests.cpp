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

#ifndef LIBND4J_SESSIONLOCALTESTS_H
#define LIBND4J_SESSIONLOCALTESTS_H

#include "testlayers.h"
#include <NDArrayFactory.h>
#include <graph/SessionLocalStorage.h>

using namespace nd4j::graph;

class SessionLocalTests : public testing::Test {
public:

};

TEST_F(SessionLocalTests, BasicTests_1) {
    VariableSpace variableSpace;
    SessionLocalStorage storage(&variableSpace, nullptr);

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.startSession();
    }

    ASSERT_EQ(4, storage.numberOfSessions());

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.endSession();
    }

    ASSERT_EQ(0, storage.numberOfSessions());
}


TEST_F(SessionLocalTests, BasicTests_2) {
    VariableSpace variableSpace;
    SessionLocalStorage storage(&variableSpace, nullptr);
    auto alpha = nd4j::NDArrayFactory::create_<float>('c',{5,5});
    alpha->assign(0.0);

    variableSpace.putVariable(-1, alpha);

#pragma omp parallel for num_threads(4)
    for (int e = 0; e < 4; e++) {
        storage.startSession();

        auto varSpace = storage.localVariableSpace();

        auto arr = varSpace->getVariable(-1)->getNDArray();
        arr->applyScalar(nd4j::scalar::Add, (float) e+1);
    }

    float lastValue = 0.0f;
    for (int e = 1; e <= 4; e++) {
        auto varSpace = storage.localVariableSpace((Nd4jLong) e);

        auto arr = varSpace->getVariable(-1)->getNDArray();

        //nd4j_printf("Last value: %f; Current value: %f\n", lastValue, arr->e(0));

        ASSERT_NE(lastValue, arr->e<float>(0));
        lastValue = arr->e<float>(0);
    }
}

#endif //LIBND4J_SESSIONLOCALTESTS_H
