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
#include <array/NDArray.h>
#include <flatbuffers/flatbuffers.h>
#include <graph/Graph.h>
#include <graph/GraphUtils.h>
#include <graph/Node.h>
#include <graph/cpu/FunctionalReplayHandle.h>
#include <graph/scheme/graph_generated.h>
#include <graph/scheme/node_generated.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/DeclarableOp.h>

#include <cstdint>
#include <utility>
#include <vector>

#include <ops/declarable/generic/parity_ops.cpp>

#include "testlayers.h"

using namespace sd;
using namespace sd::graph;

class GraphTests : public NDArrayTests {
 public:
  GraphTests() {
  }
};

namespace {

struct FunctionalReplayTestState {
  std::vector<int> slots;
  std::vector<FunctionalReplayCommandType> types;
  int failSlot = -1;
};

Status executeRecordedCommand(
    void* userData, const FunctionalReplayCommand& command) {
  auto* state = static_cast<FunctionalReplayTestState*>(userData);
  state->slots.push_back(command.slotIndex);
  state->types.push_back(command.type);
  return command.slotIndex == state->failSlot
             ? Status::KERNEL_FAILURE
             : Status::OK;
}

FunctionalReplayExecutionContext contextFor(
    FunctionalReplayTestState* state) {
  FunctionalReplayExecutionContext context;
  context.userData = state;
  context.execute = executeRecordedCommand;
  return context;
}

FunctionalReplayPointerBinding pointerBinding(
    FunctionalReplayPointerRole role, int index, int sourceType,
    bool requiredAtEntry, uintptr_t identitySeed, LongType offset = 0,
    bool live = true) {
  FunctionalReplayPointerBinding binding;
  binding.role = role;
  binding.index = index;
  binding.sourceType = sourceType;
  binding.requiredAtEntry = requiredAtEntry;
  binding.array = reinterpret_cast<const void*>(identitySeed + 0x10);
  binding.dataBuffer = reinterpret_cast<const void*>(identitySeed + 0x20);
  binding.primaryBuffer = reinterpret_cast<const void*>(identitySeed + 0x30);
  binding.specialBuffer = reinterpret_cast<const void*>(identitySeed + 0x40);
  binding.shapeInfo = reinterpret_cast<const void*>(identitySeed + 0x50);
  binding.offset = offset;
  binding.length = 4;
  binding.dataType = 5;
  binding.live = live;
  return binding;
}

}  // namespace

TEST_F(GraphTests, FunctionalReplayExplicitFactoryCreatesFunctionalBackend) {
  auto handle = GraphReplayFactory::createFunctional();

  ASSERT_NE(nullptr, handle.get());
  ASSERT_STREQ("Functional", handle->backendName());
  ASSERT_EQ(ReplayState::EMPTY, handle->getState());
}

TEST_F(GraphTests, ReplayCapabilityMatrixRequiresBothHandleAndRecorder) {
  ReplayCapabilityMatrix matrix;
  matrix.functional = {true, true};
  matrix.hip = {true, false};

  ASSERT_TRUE(matrix.canCreate(ReplayBackend::HIP));
  ASSERT_FALSE(matrix.canExecute(ReplayBackend::HIP));
  ASSERT_FALSE(matrix.hasExecutableHardwareReplay());
  ASSERT_EQ(ReplayBackend::FUNCTIONAL, matrix.preferredExecutable());

  matrix.hip.recorderAvailable = true;
  ASSERT_TRUE(matrix.canExecute(ReplayBackend::HIP));
  ASSERT_TRUE(matrix.hasExecutableHardwareReplay());
  ASSERT_EQ(ReplayBackend::HIP, matrix.preferredExecutable());
}

TEST_F(GraphTests, ReplayCapabilityMatrixUsesStablePortablePriority) {
  ReplayCapabilityMatrix matrix;
  matrix.functional = {true, true};
  matrix.metal = {true, true};
  matrix.vulkan = {true, true};
  matrix.cuda = {true, true};

  ASSERT_EQ(ReplayBackend::CUDA, matrix.preferredExecutable());
  matrix.cuda.recorderAvailable = false;
  ASSERT_EQ(ReplayBackend::VULKAN, matrix.preferredExecutable());
  matrix.vulkan.recorderAvailable = false;
  ASSERT_EQ(ReplayBackend::METAL, matrix.preferredExecutable());
  matrix.metal.recorderAvailable = false;
  ASSERT_EQ(ReplayBackend::FUNCTIONAL, matrix.preferredExecutable());
}

TEST_F(GraphTests, CurrentReplayFactoryNeverSelectsHandleOnlyScaffolds) {
  auto matrix = GraphReplayFactory::capabilities();
  auto preferred = matrix.preferredExecutable();

  ASSERT_TRUE(preferred == ReplayBackend::NONE ||
              matrix.canExecute(preferred));
  if (matrix.hip.handleAvailable && !matrix.hip.recorderAvailable) {
    ASSERT_NE(ReplayBackend::HIP, preferred);
  }
  if (matrix.levelZero.handleAvailable &&
      !matrix.levelZero.recorderAvailable) {
    ASSERT_NE(ReplayBackend::LEVEL_ZERO, preferred);
  }
  if (matrix.metal.handleAvailable && !matrix.metal.recorderAvailable) {
    ASSERT_NE(ReplayBackend::METAL, preferred);
  }
  if (matrix.tpu.handleAvailable && !matrix.tpu.recorderAvailable) {
    ASSERT_NE(ReplayBackend::TPU, preferred);
  }
  if (matrix.hexagon.handleAvailable && !matrix.hexagon.recorderAvailable) {
    ASSERT_NE(ReplayBackend::HEXAGON, preferred);
  }
}

TEST_F(GraphTests, FunctionalReplayRecordsAndExecutesSemanticProgramInOrder) {
  FunctionalReplayHandle handle;
  sd::ops::identity executeOp;
  sd::ops::identity identityOp;
  sd::ops::identity batchedOp;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.recordOp(&executeOp, 2));
  ASSERT_TRUE(handle.recordIdentity(&identityOp, 3));
  ASSERT_TRUE(handle.recordBatchedGemm(&batchedOp, 7, 4));
  ASSERT_TRUE(handle.endCapture(nullptr));
  ASSERT_TRUE(handle.finalize());
  ASSERT_TRUE(handle.hasReplayProgram());
  ASSERT_FALSE(handle.hasPointerSnapshot());

  FunctionalReplayTestState state;
  auto context = contextFor(&state);
  GraphReplayHandle* genericHandle = &handle;
  ASSERT_FALSE(genericHandle->replay(nullptr));
  ASSERT_EQ(Status::OK, handle.replayWithContext(context));

  ASSERT_EQ((std::vector<int>{2, 3, 7}), state.slots);
  ASSERT_EQ(3, static_cast<int>(state.types.size()));
  ASSERT_EQ(FunctionalReplayCommandType::EXECUTE_SLOT, state.types[0]);
  ASSERT_EQ(FunctionalReplayCommandType::FORWARD_IDENTITY, state.types[1]);
  ASSERT_EQ(FunctionalReplayCommandType::EXECUTE_BATCHED_GEMM, state.types[2]);

  auto stats = handle.getStatistics();
  ASSERT_EQ(3, stats.numOperations);
  ASSERT_EQ(1, stats.replayCount);
  ASSERT_GE(stats.captureTimeMs, 0.0);
  ASSERT_GE(stats.lastReplayTimeMs, 0.0);
}

TEST_F(GraphTests, FunctionalReplayRejectsInvalidRecordingWithoutPublishingIt) {
  FunctionalReplayHandle handle;
  sd::ops::identity firstOp;
  sd::ops::identity duplicateOp;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.recordOp(&firstOp, 4));
  ASSERT_FALSE(handle.recordOp(&duplicateOp, 4));
  ASSERT_EQ(ReplayState::ERRORED, handle.getState());
  ASSERT_FALSE(handle.endCapture(nullptr));
  ASSERT_FALSE(handle.hasReplayProgram());
  ASSERT_EQ(0, handle.getStatistics().numOperations);

  handle.abortCapture();
  ASSERT_EQ(ReplayState::EMPTY, handle.getState());
}

TEST_F(GraphTests, FunctionalReplayFailedReplayDoesNotCommitStatistics) {
  FunctionalReplayHandle handle;
  sd::ops::identity firstOp;
  sd::ops::identity failingOp;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.recordOp(&firstOp, 1));
  ASSERT_TRUE(handle.recordOp(&failingOp, 5));
  ASSERT_TRUE(handle.endCapture(nullptr));
  ASSERT_TRUE(handle.finalize());

  FunctionalReplayTestState state;
  state.failSlot = 5;
  auto context = contextFor(&state);
  ASSERT_EQ(Status::KERNEL_FAILURE, handle.replayWithContext(context));
  ASSERT_EQ((std::vector<int>{1, 5}), state.slots);
  ASSERT_EQ(0, handle.getStatistics().replayCount);
  ASSERT_EQ(ReplayState::READY, handle.getState());
}

TEST_F(GraphTests, FunctionalReplayAbortRestoresPreviouslyPublishedProgram) {
  FunctionalReplayHandle handle;
  sd::ops::identity publishedOp;
  sd::ops::identity replacementOp;
  sd::ops::identity outOfOrderOp;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.recordOp(&publishedOp, 1));
  ASSERT_TRUE(handle.endCapture(nullptr));
  ASSERT_TRUE(handle.finalize());
  double publishedCaptureTime = handle.getStatistics().captureTimeMs;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.recordOp(&replacementOp, 8));
  ASSERT_FALSE(handle.recordOp(&outOfOrderOp, 7));
  handle.abortCapture();

  ASSERT_EQ(ReplayState::READY, handle.getState());
  ASSERT_EQ(publishedCaptureTime, handle.getStatistics().captureTimeMs);
  FunctionalReplayTestState state;
  auto context = contextFor(&state);
  ASSERT_EQ(Status::OK, handle.replayWithContext(context));
  ASSERT_EQ((std::vector<int>{1}), state.slots);
}

TEST_F(GraphTests, FunctionalReplayFinalizedZeroCommandProgramIsReplayable) {
  FunctionalReplayHandle handle;

  ASSERT_TRUE(handle.beginCapture(nullptr));
  ASSERT_TRUE(handle.endCapture(nullptr));
  ASSERT_TRUE(handle.finalize());
  ASSERT_TRUE(handle.hasReplayProgram());
  ASSERT_TRUE(handle.getProgram().empty());

  FunctionalReplayExecutionContext context;
  ASSERT_EQ(Status::OK, handle.replayWithContext(context));
  ASSERT_EQ(1, handle.getStatistics().replayCount);
}

TEST_F(GraphTests, FunctionalReplayPointerTrackerAcceptsCompleteLateBoundChurn) {
  FunctionalReplayPointerTracker tracker;
  std::vector<FunctionalReplayPointerBinding> captured = {
      pointerBinding(FunctionalReplayPointerRole::EXTERNAL_INPUT, 0, 2, true, 0x100),
      pointerBinding(FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT, 5, 0, true, 0x200),
      pointerBinding(FunctionalReplayPointerRole::SEGMENT_OUTPUT, 6, 0, false, 0x300),
      pointerBinding(FunctionalReplayPointerRole::SEGMENT_OUTPUT, 9, 0, false, 0x400)};

  ASSERT_EQ(Status::OK, tracker.publish(captured));
  ASSERT_TRUE(tracker.hasSnapshot());

  std::vector<FunctionalReplayPointerBinding> replayed = {
      pointerBinding(FunctionalReplayPointerRole::EXTERNAL_INPUT, 0, 2, true, 0x500),
      pointerBinding(FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT, 5, 0, true, 0x600),
      pointerBinding(FunctionalReplayPointerRole::SEGMENT_OUTPUT, 6, 0, false, 0x700, 2),
      pointerBinding(FunctionalReplayPointerRole::SEGMENT_OUTPUT, 9, 0, false, 0x800)};

  ASSERT_EQ(Status::OK, tracker.validateEntry(replayed));
  ASSERT_EQ(Status::OK, tracker.commit(replayed));

  const auto& changes = tracker.lastChanges();
  ASSERT_EQ(4, changes.bindingCount);
  ASSERT_EQ(4, changes.changedBindings);
  ASSERT_EQ(4, changes.arrayChanges);
  ASSERT_EQ(4, changes.dataBufferChanges);
  ASSERT_EQ(4, changes.primaryBufferChanges);
  ASSERT_EQ(4, changes.specialBufferChanges);
  ASSERT_EQ(4, changes.shapeInfoChanges);
  ASSERT_EQ(1, changes.offsetChanges);
  ASSERT_EQ(0, changes.metadataChanges);
  ASSERT_EQ(0, changes.invalidBindings);
  ASSERT_EQ(1, tracker.comparisonCount());
  ASSERT_EQ(4, tracker.totalChangedBindings());
  ASSERT_NE(tracker.capturedBindings()[0].array,
            tracker.currentBindings()[0].array);

  ASSERT_EQ(Status::OK, tracker.commit(replayed));
  ASSERT_EQ(0, tracker.lastChanges().changedBindings);
  ASSERT_EQ(2, tracker.comparisonCount());
  ASSERT_EQ(4, tracker.totalChangedBindings());
}

TEST_F(GraphTests, FunctionalReplayPointerTrackerRejectsInvalidBindingsAndTopologyDrift) {
  FunctionalReplayPointerTracker tracker;
  std::vector<FunctionalReplayPointerBinding> captured = {
      pointerBinding(FunctionalReplayPointerRole::EXTERNAL_INPUT, 0, 2, true, 0x100),
      pointerBinding(FunctionalReplayPointerRole::CROSS_SEGMENT_INPUT, 1, 0, true, 0x200),
      pointerBinding(FunctionalReplayPointerRole::SEGMENT_OUTPUT, 2, 0, false, 0x300)};

  ASSERT_EQ(Status::OK, tracker.publish(captured));

  auto staleOutputAtEntry = captured;
  staleOutputAtEntry[2].live = false;
  ASSERT_EQ(Status::OK, tracker.validateEntry(staleOutputAtEntry));

  auto missingRequiredInput = captured;
  missingRequiredInput[0].live = false;
  ASSERT_EQ(Status::BAD_INPUT, tracker.validateEntry(missingRequiredInput));

  auto topologyDrift = captured;
  topologyDrift[2].index = 3;
  ASSERT_EQ(Status::BAD_GRAPH, tracker.validateEntry(topologyDrift));

  auto invalidRoleContract = captured;
  invalidRoleContract[2].requiredAtEntry = true;
  ASSERT_EQ(Status::BAD_GRAPH, tracker.validateEntry(invalidRoleContract));

  auto pointerlessLiveInput = captured;
  pointerlessLiveInput[0].primaryBuffer = nullptr;
  pointerlessLiveInput[0].specialBuffer = nullptr;
  ASSERT_EQ(Status::BAD_INPUT, tracker.validateEntry(pointerlessLiveInput));

  auto invalidPostExecutionOutput = captured;
  invalidPostExecutionOutput[2].live = false;
  ASSERT_EQ(Status::BAD_INPUT, tracker.commit(invalidPostExecutionOutput));
  ASSERT_EQ(0, tracker.comparisonCount());

  auto nonCanonical = captured;
  std::swap(nonCanonical[0], nonCanonical[1]);
  ASSERT_EQ(Status::BAD_GRAPH, tracker.validateEntry(nonCanonical));
}

TEST_F(GraphTests, SingleInput1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0f);

  graph->getVariableSpace()->putVariable(-1, x);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_STRICT, transform::Cosine, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Abs, 3, {2}, {});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(3));

  auto node3 = graph->getVariableSpace()->getVariable(3)->getNDArray();

  ASSERT_NEAR(0.4161468, node3->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {3});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {3});
  auto nodeC = new Node(OpType_PAIRWISE, pairwise::Add, 3, {1, 2}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SingleInput3) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto v0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto v1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, v0);
  graph->getVariableSpace()->putVariable(-3, v1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2, 3});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Ones, 3, {1}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(1.4142135, v0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(1.0, v1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SingleInput4) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto v0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto v1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, v0);
  graph->getVariableSpace()->putVariable(-3, v1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {4, 5});

  auto nodeS = new Node(OpType_TRANSFORM_SAME, transform::Ones, 4, {3}, {-2});
  auto nodeE = new Node(OpType_TRANSFORM_SAME, transform::Identity, 5, {3}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeS);
  graph->addNode(nodeE);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(5, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(1.0, v0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.4142135, v1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput2) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto z1 = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z0);
  graph->getVariableSpace()->putVariable(-4, z1);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3});

  auto nodeT = new Node(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
  auto nodeU = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
  auto nodeV = new Node(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeT);
  graph->addNode(nodeU);
  graph->addNode(nodeV);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(6, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(-1.4142135, z0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.0, z1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, DoubleInput3) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-1.0);

  auto z0 = NDArrayFactory::create_<float>('c', {5, 5});
  auto z1 = NDArrayFactory::create_<float>('c', {5, 5});

  auto w = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z0);
  graph->getVariableSpace()->putVariable(-4, z1);
  graph->getVariableSpace()->putVariable(-5, w);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Neg, 3, {2}, {-3, 21});

  auto nodeT = new Node(OpType_TRANSFORM_SAME, transform::Abs, 11, {-2}, {12});
  auto nodeU = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 12, {11}, {13});
  auto nodeV = new Node(OpType_TRANSFORM_SAME, transform::Neg, 13, {12}, {-4, 21});

  auto nodeW = new Node(OpType_PAIRWISE, pairwise::Add, 21, {3, 13}, {22});
  auto nodeZ = new Node(OpType_TRANSFORM_SAME, transform::Abs, 22, {21}, {-5});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeT);
  graph->addNode(nodeU);
  graph->addNode(nodeV);
  graph->addNode(nodeW);
  graph->addNode(nodeZ);

  ASSERT_EQ(2, graph->rootNodes());
  ASSERT_EQ(8, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(-1.4142135, z0->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
  ASSERT_NEAR(-1.0, z1->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  ASSERT_NEAR(2.4142135, w->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, QuadInput1) {
  auto graph = new Graph();

  auto x0 = NDArrayFactory::create_<float>('c', {5, 5});
  x0->assign(0.0);

  auto x1 = NDArrayFactory::create_<float>('c', {5, 5});
  x1->assign(-1.0);

  auto x2 = NDArrayFactory::create_<float>('c', {5, 5});
  x2->assign(-2.0);

  auto x3 = NDArrayFactory::create_<float>('c', {5, 5});
  x3->assign(-3.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});
  z->assign(119.0);

  graph->getVariableSpace()->putVariable(-1, x0);
  graph->getVariableSpace()->putVariable(-2, x1);
  graph->getVariableSpace()->putVariable(-3, x2);
  graph->getVariableSpace()->putVariable(-4, x3);
  graph->getVariableSpace()->putVariable(-5, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {11});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {11});
  auto nodeC = new Node(OpType_TRANSFORM_SAME, transform::Abs, 3, {-3}, {21});
  auto nodeD = new Node(OpType_TRANSFORM_SAME, transform::Abs, 4, {-4}, {21});

  auto nodeP1 = new Node(OpType_PAIRWISE, pairwise::Add, 11, {1, 2}, {31});
  auto nodeP2 = new Node(OpType_PAIRWISE, pairwise::Add, 21, {3, 4}, {31});

  auto nodeZ = new Node(OpType_PAIRWISE, pairwise::Add, 31, {11, 21}, {-5});

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeC);
  graph->addNode(nodeD);
  graph->addNode(nodeP1);
  graph->addNode(nodeP2);
  graph->addNode(nodeZ);

  ASSERT_EQ(4, graph->rootNodes());
  ASSERT_EQ(7, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(6.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, InternalBranching1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(0.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  // 1.0
  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Ones, 1, {-1}, {11, 21});

  // -1
  auto nodeK = new Node(OpType_TRANSFORM_SAME, transform::Neg, 11, {1}, {12});

  // 2.0
  auto nodeL = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 12, {11}, {31});

  // -1
  auto nodeR = new Node(OpType_TRANSFORM_SAME, transform::Neg, 21, {1}, {22});

  // 1
  auto nodeS = new Node(OpType_TRANSFORM_SAME, transform::Neg, 22, {21}, {31});

  // 1.0
  auto nodeZ = new Node(OpType_PAIRWISE, pairwise::Add, 31, {12, 22}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeK);
  graph->addNode(nodeL);
  graph->addNode(nodeR);
  graph->addNode(nodeS);
  graph->addNode(nodeZ);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(6, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_EQ(3, nodeZ->getLayer());

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, ReductionsTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  for (int r = 0; r < x->rows(); r++) {
    for (int c = 0; c < x->columns(); c++) {
      x->p(r, c, -c);
    }
  }

  auto z = NDArrayFactory::create_<float>('c', {5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 1, {-1}, {2}, {1}, {});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(2, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(2.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, IndexReductionsTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  for (int r = 0; r < x->rows(); r++) {
    for (int c = 0; c < x->columns(); c++) {
      x->p(r, c, -c);
    }
  }

  auto z = NDArrayFactory::create_<LongType>('c', {5, 1});
  auto axis = NDArrayFactory::create_<LongType>('c', {1}, {1});
  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_INDEX_REDUCE, indexreduce::IndexMin, 1, {-1}, {2}, {1});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(2, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(4.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete axis;
}

#if 0
TEST_F(GraphTests, AutoOutput1) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_FLOAT, 0, 1, {-1}, {2});
    auto nodeB = new Node(OpType_TRANSFORM_FLOAT, 35, 2, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(2, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(1, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete outputs;
    delete graph;
}


TEST_F(GraphTests, AutoOutput2) {
    auto graph = new Graph();
    auto x = NDArrayFactory::create_<float>('c', {5, 5});
    x->assign(-2.0);

    graph->getVariableSpace()->putVariable(-1, x);

    auto nodeA = new Node(OpType_TRANSFORM_SAME, 0, 1, {-1}, {2, 3, -1});
    auto nodeB = new Node(OpType_TRANSFORM_SAME, 35, 2, {1}, {});
    auto nodeC = new Node(OpType_TRANSFORM_SAME, 6, 3, {1}, {});

    graph->addNode(nodeA);
    graph->addNode(nodeB);
    graph->addNode(nodeC);

    ASSERT_EQ(1, graph->rootNodes());
    ASSERT_EQ(3, graph->totalNodes());

    graph->buildGraph();

    ASSERT_TRUE(graph->getVariableSpace()->getVariable(-1) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(2) != nullptr);
    ASSERT_TRUE(graph->getVariableSpace()->getVariable(3) != nullptr);

    GraphExecutioner::execute(graph);

    auto outputs = graph->fetchOutputs();

    ASSERT_EQ(2, outputs->size());

    ASSERT_TRUE(outputs->at(0) != nullptr);

    ASSERT_NEAR(-1.0, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);
    ASSERT_NEAR(-2.0, outputs->at(1)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

    delete graph;
    delete outputs;
}
#endif

TEST_F(GraphTests, BroadcastTest1) {
  auto graph = new Graph();
  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(0.f);

  auto y = NDArrayFactory::create_<float>('c', {1, 5});
  for (int e = 0; e < y->columns(); e++) {
    y->p(e, (float)e + 1);
  }

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);
  graph->getVariableSpace()->putVariable(-3, z);

  auto nodeA = new Node(OpType_BROADCAST, broadcast::Subtract, 1, {-1, -2}, {2}, {1});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Neg, 2, {1}, {-3});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(3.0, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, ScalarTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, z);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {3});
  auto nodeE = new Node(OpType_SCALAR, scalar::Add, 3, {2}, {-2}, {}, 1.3f);

  graph->addNode(nodeA);
  graph->addNode(nodeB);
  graph->addNode(nodeE);

  ASSERT_EQ(1, graph->rootNodes());
  ASSERT_EQ(3, graph->totalNodes());

  GraphExecutioner::execute(graph);

  ASSERT_NEAR(2.714213, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, SymbolicLookupTest1) {
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  std::string p("phi");
  std::string t("theta");

  nodeA->setName(&p);
  nodeB->setName(&t);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  auto rX = graph->getVariableSpace()->getVariable(&a);
  auto rZ = graph->getVariableSpace()->getVariable(&o);

  std::string om("omicron");

  ASSERT_TRUE(rX->getNDArray() == vX->getNDArray());
  ASSERT_TRUE(rZ->getNDArray() == vZ->getNDArray());
  ASSERT_FALSE(graph->getVariableSpace()->hasVariable(&om));

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(2));

  GraphExecutioner::execute(graph);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&p));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(&t));

  ASSERT_NEAR(1.4142135, z->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, OutputValidation1) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(0, outputs->size());

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation2) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  graph->addOutput(-2);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(1, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation3) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_IMPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {});

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(1, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(0)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation4) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_EXPLICIT_AND_IMPLICIT;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {-2});

  graph->addOutput(-1);

  // not a typo. we want this value only once
  graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(2, outputs->size());

  ASSERT_NEAR(1.4142135, outputs->at(1)->getNDArray()->reduceNumber(reduce::Mean).e<float>(0), 1e-5);

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation5) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_SAME, transform::Sqrt, 2, {1}, {-2});

  graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(4, outputs->size());

  delete graph;
  delete outputs;
}

TEST_F(GraphTests, OutputValidation6) {
  auto graph = new Graph();

  graph->getExecutorConfiguration()->_outputMode = OutputMode_VARIABLE_SPACE;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto z = NDArrayFactory::create_<float>('c', {5, 5});

  auto vX = new Variable(x);
  auto vZ = new Variable(z);

  std::string a("alpha");
  std::string o("omega");

  vX->setName(&a);
  vZ->setName(&o);

  graph->getVariableSpace()->putVariable(-1, vX);
  graph->getVariableSpace()->putVariable(-2, vZ);

  auto nodeA = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeB = new Node(OpType_TRANSFORM_FLOAT, transform::Sqrt, 2, {1}, {});

  // graph->addOutput(-1);

  graph->addNode(nodeA);
  graph->addNode(nodeB);

  GraphExecutioner::execute(graph);

  auto outputs = graph->fetchOutputs();

  ASSERT_EQ(4, outputs->size());
  delete graph;
  delete outputs;
}

TEST_F(GraphTests, TestMultiOutput1) {
  ops::testop2i2o op1;
  auto graph = new Graph();

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  auto y = NDArrayFactory::create_<float>('c', {5, 5});
  y->assign(-3.0);

  graph->getVariableSpace()->putVariable(-1, x);
  graph->getVariableSpace()->putVariable(-2, y);

  // Abs
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {11});
  nodeA0->markInplace(false);
  auto nodeB0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {-2}, {11});
  nodeB0->markInplace(false);

  auto op = ops::OpRegistrator::getInstance().getOperation("testop2i2o");

  // this op will add 1.0 to first input, and 2.0 for second input
  auto nodeT = new Node(op, 11, {1, 2}, {21, 31}, {}, 0.0f);
  nodeT->setName("TestOp2i2o");
  nodeT->markInplace(false);

  // this op will subtract this value from 1.0
  auto nodeX = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 21);
  nodeX->markInplace(false);
  nodeX->pickInput(11, 0);

  // this op will subtract this value from 1.0
  auto nodeY = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 31);
  nodeY->markInplace(false);
  nodeY->pickInput(11, 1);

  graph->addNode(nodeA0);
  graph->addNode(nodeB0);
  graph->addNode(nodeT);
  graph->addNode(nodeX);
  graph->addNode(nodeY);

  std::pair<int, int> pair0(11, 0);
  std::pair<int, int> pair1(11, 1);

  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair0));
  ASSERT_TRUE(graph->getVariableSpace()->hasVariable(pair1));

  Status status = GraphExecutioner::execute(graph);

  ASSERT_EQ(sd::Status::OK, status);

  ASSERT_NEAR(-2.0f, graph->getVariableSpace()->getVariable(21)->getNDArray()->meanNumber().e<float>(0), 1e-5);
  ASSERT_NEAR(-4.0f, graph->getVariableSpace()->getVariable(31)->getNDArray()->meanNumber().e<float>(0), 1e-5);

  delete graph;
}

TEST_F(GraphTests, TestDivergentNode1) {
  auto op = ops::OpRegistrator::getInstance().getOperation("Switch");
  auto nodeY = new Node(op, 1);

  ASSERT_TRUE(nodeY->isDivergencePoint());
  ASSERT_TRUE(nodeY->isActive());

  delete nodeY;
}

TEST_F(GraphTests, MemoryEstimationTest1) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);

  ASSERT_EQ(2, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(25 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest2) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);

  ASSERT_EQ(2, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(0, memReq);
}

TEST_F(GraphTests, MemoryEstimationTest3) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 3, {2}, {}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(26 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest4) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(OpType_REDUCE_FLOAT, reduce::Mean, 3, {2}, {}, {1});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ(30 * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, MemoryEstimationTest5) {
  Graph graph;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-2.0);

  graph.getVariableSpace()->putVariable(-1, x);

  ops::testcustom op;

  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {3});
  auto nodeA2 = new Node(&op, 3, {2}, {}, {});
  nodeA1->markInplace(false);

  graph.addNode(nodeA0);
  graph.addNode(nodeA1);
  graph.addNode(nodeA2);

  graph.buildGraph();

  ASSERT_EQ(3, graph.totalNodes());
  ASSERT_EQ(1, graph.rootNodes());

  auto memReq = graph.estimateRequiredMemory();

  ASSERT_EQ((25 + 100) * x->sizeOfT(), memReq);
}

TEST_F(GraphTests, TestGraphInGraph_1) {
  // this one is external graph
  Graph graphA;

  // and this ons is embedded
  Graph graphB;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-5.0);

  auto modifier = NDArrayFactory::create_<float>('c', {5, 5});
  modifier->assign(3.0);

  graphA.getVariableSpace()->putVariable(-1, x);
  graphB.getVariableSpace()->putVariable(-2, modifier);

  // this is placeholder variable
  graphB.getVariableSpace()->putVariable(-1, new Variable(true));

  // abs, result is 5
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  // 1-, result -4
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 2, {1}, {3});

  // graph should return 12: abs(3.0 x -4)
  auto nodeA2 = new Node(OpType_GRAPH, -1, 3, {2}, {4});

  // 1 - 12 = -11
  auto nodeA3 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 4, {3}, {});

  nodeA2->setGraph(&graphB);

  graphA.addNode(nodeA0);
  graphA.addNode(nodeA1);
  graphA.addNode(nodeA2);
  graphA.addNode(nodeA3);

  // this is going to be PWT
  auto nodeB0 = new Node(OpType_PAIRWISE, pairwise::Multiply, 1, {-1, -2}, {2});
  auto nodeB1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graphB.addNode(nodeB0);
  graphB.addNode(nodeB1);

  graphB.buildGraph();
  graphA.buildGraph();

  ASSERT_EQ(0, nodeA0->getLayer());
  ASSERT_EQ(1, nodeA1->getLayer());
  ASSERT_EQ(2, nodeA2->getLayer());
  ASSERT_EQ(3, nodeA3->getLayer());

  ASSERT_EQ(0, nodeB0->getLayer());
  ASSERT_EQ(1, nodeB1->getLayer());

  Status status = GraphExecutioner::execute(&graphA);
  ASSERT_EQ(sd::Status::OK, status);

  float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber().e<float>(0);


  ASSERT_NEAR(-11.0, m, 1e-5);
}

// test for symbolic lookup
TEST_F(GraphTests, TestGraphInGraph_2) {
  // this one is external graph
  Graph graphA;

  // and this ons is embedded
  Graph graphB;

  auto x = NDArrayFactory::create_<float>('c', {5, 5});
  x->assign(-5.0);

  auto modifier = NDArrayFactory::create_<float>('c', {5, 5});
  modifier->assign(3.0);

  std::string nameA1("_nodeA1");

  graphA.getVariableSpace()->putVariable(-1, x);
  graphB.getVariableSpace()->putVariable(-2, modifier);

  // this is placeholder variable
  auto placeHolder = new Variable(true);
  placeHolder->setName(&nameA1);
  graphB.getVariableSpace()->putVariable(-1, placeHolder);

  // abs, result is 5
  auto nodeA0 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 1, {-1}, {2});
  // 1-, result -4
  auto nodeA1 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 2, {1}, {3});
  nodeA1->setName(nameA1);

  // graph should return 12: abs(3.0 x -4)
  auto nodeA2 = new Node(OpType_GRAPH, -1, 3, {2}, {4});

  // 1 - 12 = -11
  auto nodeA3 = new Node(OpType_TRANSFORM_SAME, transform::OneMinus, 4, {3}, {});

  nodeA2->setGraph(&graphB);

  graphA.addNode(nodeA0);
  graphA.addNode(nodeA1);
  graphA.addNode(nodeA2);
  graphA.addNode(nodeA3);

  // this is going to be PWT
  auto nodeB0 = new Node(OpType_PAIRWISE, pairwise::Multiply, 1, {-1, -2}, {2});
  auto nodeB1 = new Node(OpType_TRANSFORM_SAME, transform::Abs, 2, {1}, {});

  graphB.addNode(nodeB0);
  graphB.addNode(nodeB1);

  graphB.buildGraph();
  graphA.buildGraph();

  ASSERT_EQ(0, nodeA0->getLayer());
  ASSERT_EQ(1, nodeA1->getLayer());
  ASSERT_EQ(2, nodeA2->getLayer());
  ASSERT_EQ(3, nodeA3->getLayer());

  ASSERT_EQ(0, nodeB0->getLayer());
  ASSERT_EQ(1, nodeB1->getLayer());

  Status status = GraphExecutioner::execute(&graphA);
  ASSERT_EQ(sd::Status::OK, status);

  float m = graphA.getVariableSpace()->getVariable(4)->getNDArray()->meanNumber().e<float>(0);


  ASSERT_NEAR(-11.0, m, 1e-5);
}


TEST_F(GraphTests, Test_Inplace_Outputs_1) {
  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto z = NDArrayFactory::create<float>('c', {2, 3});

  ops::test_output_reshape op;
  auto result = op.execute({&x}, {&z}, {}, {}, {});
  ASSERT_EQ(sd::Status::OK, result);

  ASSERT_EQ(exp,z);
}

TEST_F(GraphTests, Test_Inplace_Outputs_2) {
#ifndef __APPLE_OS__
  // we dont want testing this on apple. due to try/catch

  auto x = NDArrayFactory::create<float>('c', {2, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto exp = NDArrayFactory::create<float>('c', {6}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto z = NDArrayFactory::create<float>('c', {3, 3});

  bool failed = false;
  ops::test_output_reshape op;
  try {
    op.execute({&x}, {&z}, {}, {}, {});

  } catch (const std::runtime_error& e) {
    failed = true;
  }

  ASSERT_TRUE(failed);
#endif
}



TEST_F(GraphTests, Test_Minifier_2) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
  ASSERT_EQ(0, GraphUtils::runPreprocessor("../include/ops/specials.h", "libnd4j_mini2.hpp"));
  // remove file from filesystem
#ifdef __linux__
  ASSERT_EQ(0, unlink("libnd4j_mini2.hpp"));
#endif
}

TEST_F(GraphTests, Test_Minifier_3) {
  // run preprocessor to produce single header
  // if all ok - return value is 0, if error - non-zero value will be returned
#ifdef __linux__
  ASSERT_EQ(0x100, GraphUtils::runPreprocessor("/include/ops/ops.h", "libnd4j_mini3.hpp"));
#endif

}
