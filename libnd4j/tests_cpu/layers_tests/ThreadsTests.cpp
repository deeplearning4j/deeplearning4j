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
#include <loops/type_conversions.h>
#include <execution/Threads.h>
#include <chrono>
#include <execution/ThreadPool.h>

using namespace nd4j;
using namespace nd4j::ops;
using namespace nd4j::graph;

class ThreadsTests : public testing::Test {
public:

};

TEST_F(ThreadsTests, basic_test_1) {
    if (!Environment::getInstance()->isCPU())
        return;

    auto instance = samediff::ThreadPool::getInstance();

    auto array = NDArrayFactory::create<float>('c', {512, 768});
    auto like = array.like();
    auto buffer = array.bufferAsT<float>();
    auto lbuffer = like.bufferAsT<float>();

    auto func = PRAGMA_THREADS_FOR {
        PRAGMA_OMP_SIMD
        for (uint64_t e = start; e < stop; e += increment) {
            buffer[e] += 1.0f;
        }
    };

    auto timeStartThreads = std::chrono::system_clock::now();
    samediff::Threads::parallel_for(func, 6, 0, array.lengthOf(), 1);
    auto timeEndThreads = std::chrono::system_clock::now();
    auto outerTimeThreads = std::chrono::duration_cast<std::chrono::microseconds> (timeEndThreads - timeStartThreads).count();

    auto timeStartOmp = std::chrono::system_clock::now();
    PRAGMA_OMP_PARALLEL_FOR_SIMD
    for (uint64_t e = 0; e < array.lengthOf(); e ++) {
        lbuffer[e] += 1.0f;
    }
    auto timeEndOmp = std::chrono::system_clock::now();
    auto outerTimeOmp = std::chrono::duration_cast<std::chrono::microseconds> (timeEndOmp - timeStartOmp).count();

    ASSERT_NEAR((float) array.lengthOf(), array.sumNumber().e<float>(0), 1e-5f);

    nd4j_printf("Threads time: %lld us; OMP time: %lld us; %p\n", outerTimeThreads, outerTimeOmp, instance)
}