/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
/*
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
 */
/*
TEST_F(PlaygroundTests, test_s_4) {
    std::atomic<float> f;
    std::atomic<int> s;
    std::vector<Nd4jLong> valuesX, valuesY;
    int iterations = 1000;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);

    ////////

    auto x = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    auto z = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    x.linspace(1.0);

    auto xs0 = x.sizeAt(0);
    auto xs1 = x.sizeAt(1);
    auto xs2 = x.sizeAt(2);
    auto xs3 = x.sizeAt(3);

    auto buffer = x.bufferAsT<float>();
    auto zbuffer = z.bufferAsT<float>();

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int i = 0; i < xs0; i++) {
            for (int j = 0; j < xs1; j++) {
                auto thread_id = omp_get_thread_num();
                for (int k = 0; k < xs2; k++) {
                    for (int l = 0; l < xs3; l++) {
                        zbuffer[thread_id] += buffer[i * j + (k*l)] * 2.5f;
                    }
                }
            }
        }
        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesX.emplace_back(outerTime);
    }


    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        auto f2d = PRAGMA_THREADS_FOR_2D {
            for (auto i = start_x; i < stop_x; i++) {
                for (auto j = start_y; j < stop_y; j++) {

                    for (auto k = 0; k < xs2; k++) {
                        for (auto l = 0; l < xs3; l++) {
                            zbuffer[thread_id] += buffer[i * j + (k * l)] * 2.5f;
                        }
                    }
                }
            }
        };
        samediff::Threads::parallel_for(f2d, 0, xs0, 1, 0, xs1, 1);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesY.emplace_back(outerTime);
    }

    if (valuesX.size() > 0) {
        std::sort(valuesX.begin(), valuesX.end());
        nd4j_printf("OpenMP time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0], valuesX[valuesX.size() - 1]);
    }

    if (valuesY.size() > 0) {
        std::sort(valuesY.begin(), valuesY.end());
        nd4j_printf("Threads time: %lld; Min: %lld; Max: %lld;\n", valuesY[valuesY.size() / 2], valuesY[0], valuesY[valuesY.size() - 1]);
    }

    nd4j_printf("Sum: %f\n", z.sumNumber().e<float>(0));
}


TEST_F(PlaygroundTests, test_s_5) {
    auto x = NDArrayFactory::create<float>('c', {32, 1, 28, 28});

    std::vector<Nd4jLong> values;
    auto iterations = 100;

    auto startX = 0;
    auto stopX = x.sizeAt(0);
    auto incX = 1;
    auto startY = 0;
    auto stopY = x.sizeAt(1);
    auto incY = 1;
    auto numThreads = 4;

    // number of elements per loop
    auto delta_x = (stopX - startX);
    auto delta_y = (stopY - startY);

    // number of iterations per loop
    auto itersX = delta_x / incX;
    auto itersY = delta_y / incY;

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        auto splitLoop = samediff::ThreadsHelper::pickLoop2d(numThreads, itersX, itersY);
        auto span = samediff::Span2::build(splitLoop, 0, numThreads, startX, stopX, incX, startY, stopY, incY);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
}


TEST_F(PlaygroundTests, test_s_6) {
    auto x = NDArrayFactory::create<float>('c', {1024 * 1024 * 64});
    auto buffer = x.bufferAsT<float>();
    auto len = x.lengthOf();
    std::vector<Nd4jLong> values;
    auto iterations = 1000;

    for (int i = 0; i < iterations; i++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        for (int e = 0; e < len; e++) {
            buffer[e] = (buffer[e] + 1.72f) * 3.17f - 0.0012f;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
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
 */

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

//////////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, my) {

    int bS=1, iH=56,iW=56,  iC=144,mC=1,  kH=3,kW=3,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oC=iC*mC;
    int       oH=56,oW=56;
    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<float>('c', {bS, iH, iW, iC});
    auto weights  = NDArrayFactory::create<float>('c', {kH, kW, iC, mC});

    input = 2.;
    weights.linspace(0.1, 0.1);

    nd4j::ops::depthwise_conv2d op;
    auto results = op.execute({&input, &weights}, {}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

    delete results;
}

*/


