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
#include <array/ArrayOptions.h>
#include <execution/AffinityManager.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/broadcastable.h>
#include <helpers/MmulHelper.h>
#include <thread>


using namespace sd;

class MultiDeviceTests : public testing::Test {
public:

};

void createArrays(int limit, std::vector<NDArray*> &arrays) {
    auto deviceId = AffinityManager::currentDeviceId();
    auto numDevices = AffinityManager::numberOfDevices();

    for (int e = 0; e < limit; e++) {
        auto value = deviceId * limit + e;
        arrays[value] = NDArrayFactory::create_<float>('c', {10});
        arrays[value]->assign(value);
        //nd4j_printf("device_%i; value: [%i]; mean: [%f]\n", deviceId, value, arrays[value]->meanNumber().e<float>(0));
    }
}

TEST_F(MultiDeviceTests, test_multi_device_migration_1) {
    auto deviceId = AffinityManager::currentDeviceId();
    auto numDevices = AffinityManager::numberOfDevices();
    auto numArrays = 10;
    std::vector<NDArray*> arrays(numDevices * numArrays);

    // filling list of arrays on multiple threads
    for (int e = 0; e < numDevices; e++) {
        std::thread t1(createArrays, numArrays, std::ref(arrays));

        t1.join();
    }

    // at this moment all arrays are build, so we can test migration
    for (int e = 0; e < arrays.size(); e++) {
        ASSERT_NEAR((float) e, arrays[e]->meanNumber().e<float>(0), 1e-5f);
        delete arrays[e];
    }
}
