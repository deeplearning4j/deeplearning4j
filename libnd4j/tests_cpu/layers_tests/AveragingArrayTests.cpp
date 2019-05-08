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
#include <AveragingArrayProxy.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class AveragingArrayTests : public testing::Test {
public:

};

TEST_F(AveragingArrayTests, test_basic_reads_1) {
    auto exp0 = NDArrayFactory::create<double>('c', {1, 5},{3.0, 3.0, 3.0, 3.0, 3.0});

    auto original = NDArrayFactory::create<double>('c', {100, 5});
    original.assign(1.0);

    AveragingArrayProxy proxy(&original);

    auto writeable0 = proxy.writeable(1, 0);
    auto writeable1 = proxy.writeable(1, 1);
    auto writeable2 = proxy.writeable(2, 1);

    ASSERT_FALSE(writeable0 == nullptr);
    ASSERT_FALSE(writeable1 == nullptr);
    ASSERT_FALSE(writeable2 == nullptr);

    writeable0->assign(2.0);
    writeable1->assign(4.0);
    writeable2->assign(3.0);

    auto r = proxy.collapseWrites();

    ASSERT_TRUE(r);

    auto row1 = original({1,2, 0,0}, true);
    auto row2 = original({2,3, 0,0}, true);

    ASSERT_EQ(exp0, row1);
    ASSERT_EQ(exp0, row2);
}
