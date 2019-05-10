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
// Created by raver119 on 11.10.2017.
//

#include <memory/MemoryReport.h>
#include <memory/MemoryUtils.h>
#include "testlayers.h"

using namespace nd4j::memory;

class MemoryUtilsTests : public testing::Test {
public:

};

TEST_F(MemoryUtilsTests, BasicRetrieve_1) {
    MemoryReport reportA;
    MemoryReport reportB;

#ifdef _WIN32
    if (1 > 0)
        return;
#endif


    MemoryUtils::retrieveMemoryStatistics(reportA);


    ASSERT_NE(reportA, reportB);
}
