/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <execution/ThreadPool.h>
#include <graph/Graph.h>
#include <graph/Node.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <helpers/BenchmarkHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/GradCheck.h>
#include <helpers/Loops.h>
#include <helpers/MmulHelper.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/RandomLauncher.h>
#include <helpers/threshold.h>
#include <loops/type_conversions.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/im2col.h>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/scatter.h>
#include <ops/ops.h>

#include <array>
#include <chrono>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class PerformanceTests : public testing::Test {
 public:
  int numIterations = 100;

  PerformanceTests() { samediff::ThreadPool::getInstance(); }
};

#ifdef RELEASE_BUILD

TEST_F(PerformanceTests, test_matmul_c_f_1) {
  int iterations = 500;
  std::vector<ino64_t> valuesC, valuesF;
  for (int e = 0; e < iterations; e++) {
    auto xc = NDArrayFactory::create<float>('c', {512, 2048});
    auto yc = NDArrayFactory::create<float>('c', {2048, 512});
    auto zc = NDArrayFactory::create<float>('c', {512, 512});

    auto xf = NDArrayFactory::create<float>('f', {512, 2048});
    auto yf = NDArrayFactory::create<float>('f', {2048, 512});
    auto zf = NDArrayFactory::create<float>('f', {512, 512});

    auto warm = xc.like();
    warm.linspace(1.0);

    // zc.linspace(1.0);
    // zf.linspace(1.0);

    sd::ops::matmul op;

    auto timeStartF = std::chrono::system_clock::now();

    op.execute({&xf, &yf}, {&zf});

    auto timeEndF = std::chrono::system_clock::now();
    auto outerTimeF = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEndF - timeStartF).count();

    auto timeStartC = std::chrono::system_clock::now();

    op.execute({&xc, &yc}, {&zc});

    auto timeEndC = std::chrono::system_clock::now();
    auto outerTimeC = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEndC - timeStartC).count();

    valuesF.emplace_back(outerTimeF);
    valuesC.emplace_back(outerTimeC);
  }

  std::sort(valuesC.begin(), valuesC.end());
  std::sort(valuesF.begin(), valuesF.end());

  sd_printf("Median time C: [%lld]; Median time F: [%lld];", valuesC[valuesC.size() / 2], valuesF[valuesF.size() / 2]);
}

TEST_F(PerformanceTests, test_maxpooling2d_1) {
  std::vector<sd::LongType> valuesX;
  // auto x = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
  // auto z = NDArrayFactory::create<float>('c', {32, 3, 224, 224});
  auto x = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
  auto z = NDArrayFactory::create<float>('c', {8, 3, 64, 64});
  x.linspace(1.0f);
  sd::LongType k = 5;

  sd::LongType iArgs[]{k, k, 1, 1, 0, 0, 1, 1, 1};
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

    if ((i + 1) % 1000 == 0) sd_printf("Iteration %i finished...\n", i + 1);
  }

  std::sort(valuesX.begin(), valuesX.end());
  sd_printf("Execution time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0],
            valuesX[valuesX.size() - 1]);
}

#endif
