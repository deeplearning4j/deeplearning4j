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
// Created by raver119 on 15.12.17.
//
#include <graph/Graph.h>
#include <graph/Node.h>
#include <helpers/OpTracker.h>
#include <ops/declarable/CustomOperations.h>

#include <chrono>

#include "testlayers.h"

using namespace sd;
using namespace sd::ops;
using namespace sd::graph;

class OpTrackerTests : public NDArrayTests {
 public:
  int numIterations = 10;
  int poolSize = 10;

  OpTrackerTests() {
    printf("\n");
    fflush(stdout);
  }
};

TEST_F(OpTrackerTests, Test_Existence_1) {
  sd::_loader loader;


  ASSERT_TRUE(OpTracker::getInstance().totalGroups() > 0);
  ASSERT_TRUE(OpTracker::getInstance().totalOperations() > 0);

  OpTracker::getInstance().exportOperations();
}

TEST_F(OpTrackerTests, Test_Ops_List_1) {
  sd::ops::less op;
  auto vec = OpRegistrator::getInstance().getAllHashes();

  for (const auto &v : vec) {
    if (v == 5484196977525668316L) {
      auto op = OpRegistrator::getInstance().getOperation(v);
    }
  }
}
