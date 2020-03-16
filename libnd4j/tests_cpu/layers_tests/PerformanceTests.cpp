/*******************************************************************************
 * Copyright (c) 2019 Konduit
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
#include <graph/Graph.h>
#include <chrono>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <loops/type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <helpers/Loops.h>
#include <helpers/RandomLauncher.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <execution/ThreadPool.h>

using namespace sd;
using namespace sd::graph;

class PerformanceTests : public testing::Test {
public:
    int numIterations = 100;

    PerformanceTests() {
        samediff::ThreadPool::getInstance();
    }
};


#ifdef RELEASE_BUILD

TEST_F(PerformanceTests, test_maxpooling2d_1) {
    std::vector<Nd4jLong> valuesX;
    // auto x = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
    // auto z = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
    auto x = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
    auto z = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
    x.linspace(1.0f);
    Nd4jLong k = 5;


    Nd4jLong iArgs[] {k,k, 1,1, 0,0, 1,1, 1};
    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setOutputArray(0, &z);
    ctx.setIArguments(iArgs, 9);

    sd::ops::maxpool2d op;

    for (int i = 0; i < numIterations; i++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesX.emplace_back(outerTime);

        if ((i + 1) % 1000 == 0)
            nd4j_printf("Iteration %i finished...\n", i + 1);
    }

    std::sort(valuesX.begin(), valuesX.end());
    nd4j_printf("Execution time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0], valuesX[valuesX.size() - 1]);
}

#endif