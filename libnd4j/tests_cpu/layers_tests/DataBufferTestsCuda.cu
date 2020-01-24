/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
#include <NDArray.h>
#include <Context.h>
#include <Node.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/convolutions.h>
#include <ops/declarable/helpers/col2im.h>
#include <helpers/RandomLauncher.h>

using namespace nd4j;
using namespace nd4j::graph;
using namespace nd4j::memory;

class DataBufferTestsCuda : public testing::Test {
public:

};

TEST_F(DataBufferTestsCuda, test_alloc_limit_1) {
    auto deviceId = AffinityManager::currentDeviceId();

    auto odLimit = MemoryCounter::getInstance()->deviceLimit(deviceId);

    auto opLimit = MemoryCounter::getInstance()->groupLimit(MemoryType::HOST);
    auto osLimit = MemoryCounter::getInstance()->groupLimit(MemoryType::DEVICE);

    auto odUse = MemoryCounter::getInstance()->allocatedDevice(deviceId);

    auto opUse = MemoryCounter::getInstance()->allocatedGroup(MemoryType::HOST);
    auto osUse = MemoryCounter::getInstance()->allocatedGroup(MemoryType::DEVICE);

    auto limitSize = odUse + 150000000;
    auto allocSize = 100000000;

    MemoryCounter::getInstance()->setDeviceLimit(deviceId, odLimit + limitSize);
    MemoryCounter::getInstance()->setGroupLimit(MemoryType::HOST, opLimit + limitSize);
    MemoryCounter::getInstance()->setGroupLimit(MemoryType::DEVICE, osLimit + limitSize);

    DataBuffer buffer(allocSize, DataType::INT32, nullptr, true);

    // separately testing per-device limits and group limits
    ASSERT_EQ(odUse + allocSize, MemoryCounter::getInstance()->allocatedDevice(deviceId));
    ASSERT_EQ(opUse + allocSize, MemoryCounter::getInstance()->allocatedGroup(MemoryType::HOST));
    ASSERT_EQ(osUse + allocSize, MemoryCounter::getInstance()->allocatedGroup(MemoryType::DEVICE));

    // setting smaller limits, to make sure next allocation fails with OOM exception
    MemoryCounter::getInstance()->setDeviceLimit(deviceId, allocSize - 100);
    MemoryCounter::getInstance()->setGroupLimit(MemoryType::DEVICE, allocSize - 100);


    // this allocation should fail, since we're allocating too much
    try {
        DataBuffer bufferFailed(allocSize + 1, DataType::INT32);
        ASSERT_TRUE(false);
    } catch (allocation_exception &e) {
        // we expect exception here
    }

    //

    // restore original limits, so subsequent tests do not fail
    MemoryCounter::getInstance()->setDeviceLimit(deviceId, odLimit);
    MemoryCounter::getInstance()->setGroupLimit(MemoryType::HOST, opLimit);
    MemoryCounter::getInstance()->setGroupLimit(MemoryType::DEVICE, osLimit);
}