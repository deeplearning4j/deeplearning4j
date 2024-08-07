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
//  @author raver119@gmail.com
//
#include <graph/GraphState.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/LegacyReduceOp.h>
#include <ops/declarable/LegacyTransformOp.h>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class GraphStateTests : public NDArrayTests {
 public:
  GraphStateTests() {
    Environment::getInstance().setDebug(false);
    Environment::getInstance().setVerbose(false);
  };

  ~GraphStateTests() {
    Environment::getInstance().setDebug(false);
    Environment::getInstance().setVerbose(false);
  }
};

/*
 * PLAN:
 * Create GraphState
 * Register Scope
 * Add few Ops to it
 * Call conditional, that refers to scopes
 * Check results
 */

TEST_F(GraphStateTests, Basic_Tests_1) {
  auto state = (GraphState *)getGraphState(117L);
  ASSERT_EQ(117L, state->id());

  // this call will create scope internally
  state->registerScope(119);

  ops::add opA;
  ops::LegacyTransformSameOp opB(transform::Neg);  // simdOps::Neg

  ArgumentsList argsA;
  ArgumentsList argsB;

  state->attachOpToScope(119, 1, &opA, argsA);
  state->attachOpToScope(119, 2, &opB, argsB);

  auto scope = state->getScope(119);
  ASSERT_TRUE(scope != nullptr);
  ASSERT_EQ(2, scope->size());

  deleteGraphState(state);
}

// just separate case for doubles wrapper in NativeOps, nothing else
TEST_F(GraphStateTests, Basic_Tests_2) {
  auto state = (GraphState *)getGraphState(117L);
  ASSERT_EQ(117L, state->id());

  // this call will create scope internally
  state->registerScope(119);

  ops::add opA;
  ops::LegacyTransformSameOp opB(transform::Neg);  // simdOps::Neg

  ArgumentsList argsA;
  ArgumentsList argsB;

  state->attachOpToScope(119, 1, &opA, argsA);
  state->attachOpToScope(119, 2, &opB, argsB);

  auto scope = state->getScope(119);
  ASSERT_TRUE(scope != nullptr);
  ASSERT_EQ(2, scope->size());

  deleteGraphState(state);
}

