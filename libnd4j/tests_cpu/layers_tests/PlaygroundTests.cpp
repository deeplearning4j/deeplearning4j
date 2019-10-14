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
// Created by raver119 on 20.11.17.
//

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <OmpLaunchHelper.h>
#include <GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <Loops.h>
#include <RandomLauncher.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

#include <ops/declarable/helpers/legacy_helpers.h>

using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
    int numIterations = 3;
    int poolSize = 10;

    PlaygroundTests() {
        printf("\n");
        fflush(stdout);
    }
};

TEST_F(PlaygroundTests, test_s_1) {
    auto t = ::runLightBenchmarkSuit(true);
    delete[] t;
}

TEST_F(PlaygroundTests, test_s_2) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    std::vector<Nd4jLong> values;

    for (int e = 0; e < 100000; e++) {
        s = 0;

        auto timeStart = std::chrono::system_clock::now();
        //samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
        PRAGMA_OMP_PARALLEL_THREADS(4) {
            s++;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    };
    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld;\n", values[values.size() / 2]);
}

TEST_F(PlaygroundTests, test_s_3) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    for (int e = 0; e < 10000; e++) {

        samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    }
}

/*
TEST_F(PlaygroundTests, test_relubp_1) {
    auto x = NDArrayFactory::create<float>('c', {128, 64, 224, 224});
    auto y = x.ulike();
    auto z = x.ulike();
    RandomGenerator rng(119, 120);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &x, -1.0, 1.0);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &y, -1.0, 1.0);

    int iterations = 10;

    auto timeStart = std::chrono::system_clock::now();
    for (int e = 0; e < iterations; e++)
        ops::helpers::reluDerivative(LaunchContext::defaultContext(), &x, &y, &z);
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    auto time = (Nd4jLong) outerTime / iterations;
    auto bw = (1000000L * (float) (x.lengthOf() * x.sizeOfT()) / time) / 1024 / 1024 / 1024;

    nd4j_printf("Time: %lld; BW: %f GB/s\n", time, bw);
}
*/
