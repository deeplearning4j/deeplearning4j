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
#include <array/NDArray.h>
#include <helpers/ShapeUtils.h>
#include <loops/reduce3.h>
#include <ops/declarable/LegacyTransformOp.h>
#include <ops/declarable/LegacyPairwiseTransformOp.h>
#include <ops/declarable/LegacyScalarOp.h>
#include <ops/declarable/LegacyReduceSameOp.h>
#include <ops/declarable/LegacyReduceFloatOp.h>
#include <ops/declarable/LegacyIndexReduceOp.h>
#include <ops/declarable/LegacyBroadcastOp.h>
#include <helpers/TAD.h>
#include <helpers/ConstantTadHelper.h>
#include <thread>
#include <execution/AffinityManager.h>

using namespace sd;
using namespace sd::ops;

class LaunchContextCudaTests : public testing::Test {
    //
};


void acquireContext(int threadId, int &deviceId) {
    deviceId = AffinityManager::currentDeviceId();

    nd4j_printf("Creating thread: [%i]; assigned deviceId: [%i];\n", threadId, deviceId);

    auto lc = LaunchContext::defaultContext();
    nd4j_printf("LC: [%p]\n", lc);

    nd4j_printf("reductionPtr: [%p]; stream: [%p];\n", lc->getReductionPointer(), lc->getCudaStream());
}

TEST_F(LaunchContextCudaTests, basic_test_1) {
    int deviceA, deviceB;
    std::thread threadA(acquireContext, 0, std::ref(deviceA));
    std::thread threadB(acquireContext, 1, std::ref(deviceB));

    threadA.join();
    threadB.join();
    nd4j_printf("All threads joined\n","");

    if (AffinityManager::numberOfDevices() > 1)
        ASSERT_NE(deviceA, deviceB);
}

void fillArray(int tid, std::vector<NDArray*> &arrays) {
    auto array = NDArrayFactory::create_<int>('c', {3, 10});
    nd4j_printf("Array created on device [%i]\n", AffinityManager::currentDeviceId());
    array->assign(tid);
    arrays[tid] = array;
}

TEST_F(LaunchContextCudaTests, basic_test_2) {
    std::vector<NDArray*> arrays(2);

    std::thread threadA(fillArray, 0, std::ref(arrays));
    std::thread threadB(fillArray, 1, std::ref(arrays));

    threadA.join();
    threadB.join();

    for (int e = 0; e < 2; e++) {
        auto array = arrays[e];
        ASSERT_EQ(e, array->e<int>(0));

        delete array;
    }
}

void initAffinity(int tid, std::vector<int> &aff) {
    auto affinity = AffinityManager::currentDeviceId();
    aff[tid] = affinity;
    nd4j_printf("Thread [%i] affined with device [%i]\n", tid, affinity);
}

TEST_F(LaunchContextCudaTests, basic_test_3) {
    auto totalThreads = AffinityManager::numberOfDevices() * 4;
    nd4j_printf("Total threads: %i\n", totalThreads);
    std::vector<int> affinities(totalThreads);

    for (int e = 0; e < totalThreads; e++) {
        std::thread thread(initAffinity, e, std::ref(affinities));

        thread.join();
    }

    std::vector<int> hits(AffinityManager::numberOfDevices());
    std::fill(hits.begin(), hits.end(), 0);

    // we need to make sure all threads were attached to "valid" devices
    for (int e = 0; e < totalThreads; e++) {
        auto aff = affinities[e];
        ASSERT_TRUE(aff >= 0 && aff < AffinityManager::numberOfDevices());

        hits[aff]++;
    }

    // now we check if all devices got some threads
    for (int e = 0; e < AffinityManager::numberOfDevices(); e++) {
        ASSERT_GT(hits[e], 0);
    }
}