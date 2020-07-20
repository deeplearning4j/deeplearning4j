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
// @author raver110@gmail.com
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
#include <ops/declarable/helpers/convolutions.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <random>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/addBias.h>
#include <ops/declarable/helpers/axis.h>
#include <ops/declarable/helpers/reductions.h>
#include <helpers/LoopsCoordsHelper.h>

using namespace sd;
using namespace sd::graph;

class PrimitivesTests : public testing::Test {
 public:

  PrimitivesTests() {
  }
};

TEST_F(PrimitivesTests, test_mod_1) {
  int ix = 7;
  int iy = 3;


  auto v = simdOps::Mod<int, int, int>::op(ix, iy);

  ASSERT_EQ(7 % 3, v);
}

TEST_F(PrimitivesTests, test_mod_2) {
  float ix = 7.f;
  float iy = 3.f;


  auto e = sd::math::nd4j_fmod<float, float, float>(ix, iy);
  auto v = simdOps::Mod<float, float, float>::op(ix, iy);

  ASSERT_NEAR(e, v, 1e-5f);
}

TEST_F(PrimitivesTests, test_mod_3) {
  float ix = 7.f;
  float iy = 0.f;


  auto e = sd::math::nd4j_fmod<float, float, float>(ix, iy);
  auto v = simdOps::Mod<float, float, float>::op(ix, iy);

  // absence of SIGFPE will be a good enough
}