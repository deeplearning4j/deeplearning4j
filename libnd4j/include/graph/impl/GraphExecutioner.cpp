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
#include <graph/generated/graph_generated.h>
#include <graph/generated/node_generated.h>
#include <graph/generated/result_generated.h>

#include <graph/GraphExecutioner.h>
#include <graph/Node.h>
#include <graph/Scope.h>
#include <graph/TimeHolder.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <loops/pairwise_transform.h>
#include <loops/scalar.h>
#include <loops/transform_same.h>
#include <memory/MemoryRegistrator.h>
#include <ops/declarable/DeclarableOp.h>


#include <array/DataTypeUtils.h>
#include <exceptions/graph_execution_exception.h>
#include <exceptions/no_results_exception.h>
#include <fcntl.h>
#include <graph/ExecutionResult.h>
#include <graph/FlatUtils.h>
#include <graph/ResultWrapper.h>
#include <graph/execution/LogicExecutor.h>
#include <graph/generated/array_generated.h>
#include <helpers/BitwiseUtils.h>
#include <helpers/ShapeUtils.h>

#include <chrono>
#include <ctime>
#include <deque>

namespace sd {
namespace graph {

/**
 * This method executes given Node (as in Op within Node)
 *
 * Basically it just does DeclarableOp::execute(Block), and ops to their job. However, there are some additional
 * functionality.
 *
 * @param graph - Graph instance pointer
 * @param node - Node instance pointer, which will be executed
 * @param variableSpace - VariableSpace instance pointer - varspace specific to current Thread/Session
 * @return
 */
Status GraphExecutioner::executeFlatNode(Graph *graph, Node *node, VariableSpace *variableSpace) {
  ::graph::OpType opType = node->opType();
  int opNum = node->opNum();
  //    std::string opName = *(node->getCustomOp()->getOpName());

  if (opType == ::graph::OpType_BOOLEAN) {
    sd_debug("Executing boolean graph node_%i", node->id());
  } else if (opType == ::graph::OpType_LOGIC) {
    sd_debug("Executing logic graph node_%i", node->id());
  } else if (opType == ::graph::OpType_GRAPH) {
    sd_debug("Executing embedded graph node_%i", node->id());
  } else if (opType != ::graph::OpType_CUSTOM) {
    sd_debug("Executing node_%i{%i}\n", node->id(), opNum);
  } else {
    sd_debug("Executing node_%i{%s}\n", node->id(), node->getCustomOp()->getOpName()->c_str());
  }

  Context context(node->getContextPrototype(), variableSpace);

  if (Environment::getInstance().isDebugAndVerbose()) {
    // sd_debug("Input variables: %i\n", node->input()->size());
    printf("       Inputs: {");
    for (size_t e = 0; e < node->input()->size(); e++) {
      printf("[%i:%i]", node->input()->at(e).first, node->input()->at(e).second);

      if (e < node->input()->size() - 1) printf(", ");
    }
    printf("}\n");
    fflush(stdout);
  }

  if (node->id() == 13) sd_debug("", "");

  // if true - this is special case: Graph-in-Graph.
  if (node->hasGraphEmbedded()) {
    auto embedded = node->getGraph();

    /**
     * basically, we should do following things here:
     * 1) fill embedded graph with input variables from this graph, if anything should be filled in
     * 2) invoke embedded graph
     * 3) announce its results as corresponding output variables in current VariableSpace
     */

    // enforcing IMPLICIT mode. or not... should we try to be smarter then user?
    // embedded->getExecutorConfiguration()->_outputMode = OutputMode_IMPLICIT;

    if (node->input()->size() != static_cast<size_t>(embedded->numberOfPlaceholders())) {
      sd_debug("Placeholders amount mismatch: %i expected, and %i available\n", node->input()->size(),
               embedded->numberOfPlaceholders());
      return Status::BAD_INPUT;
    }

    // we need to propagate required variables to the embedded graph
    ResultSet deletables;
    int cnt = 0;
    for (Variable *v : *embedded->getPlaceholders()) {
      if (v->getName() != nullptr && v->getName()->size() > 0) {
        // trying symbolic lookup first
        if (variableSpace->hasVariable(v->getName())) {
          // symbolic feeder
          auto array = variableSpace->getVariable(v->getName())->getNDArray();
          auto vr = new NDArray(array->dup(array->ordering()));
          v->setNDArray(vr);
        } else {
          sd_debug("Can't find variable [%s] in parent graph...", v->getName()->c_str());
          return Status::BAD_INPUT;
        }
      } else {
        // if we're not using symbolic lookup - we'll use sequential approach then
        auto p = node->input()->at(cnt);
        auto array = variableSpace->getVariable(p)->getNDArray();
        auto vr = new NDArray(array->dup(array->ordering()));
        v->setNDArray(vr);
      }

      cnt++;
    }

    // executing embedded graph as independent one
    Status status = execute(embedded);
    if (status != Status::OK) return status;

    //  now we should migrate its results to this node, as its own outputs
    cnt = 0;
    auto outputs = embedded->fetchOutputs();

    for (auto v : *outputs) {
      NDArray *array = v->getNDArray();
      v->setNDArray(nullptr);

      std::pair<int, int> pair(node->id(), cnt++);

      auto var = variableSpace->getVariable(pair);

      var->setNDArray(array);
      var->markRemovable(true);
    }
    deletables.size();
    delete outputs;
    sd_debug("Embedded graph execution finished. %i variable(s) migrated\n", cnt);

  } else if (node->hasCustomOp()) {
    // now, if we have something to execute - lets just execute it.
    auto status = node->getCustomOp()->execute(&context);
    if (status != Status::OK) return status;

    // propagate variables
    if (node->hasExternalOutputs()) {
      for (auto v : *node->output()) {
        if (variableSpace->hasExternalVariable(v.first)) {
          variableSpace->getVariable(v.first)->getNDArray()->assign(
              variableSpace->getVariable(node->id())->getNDArray());
        }
      }
    }

    return status;
  }
  return Status::OK;
}

/**
 * This method executes given Graph instance, and returns error code.
 *
 * @param graph
 * @return one of error codes defined in pointercast.h
 */
Status GraphExecutioner::execute(Graph *graph, VariableSpace *variableSpace) {
  auto __variableSpace = variableSpace == nullptr ? graph->getVariableSpace() : variableSpace;

  bool tempFlow = false;
  if (__variableSpace->flowPath() == nullptr) {
    tempFlow = true;
    __variableSpace->setFlowPath(new FlowPath());
  }
  auto flowPath = __variableSpace->flowPath();

  LongType tb0 = Environment::getInstance().isProfiling() ? GraphProfile::currentTime() : 0L;
  graph->buildGraph();

  auto footprintForward = memory::MemoryRegistrator::getInstance().getGraphMemoryFootprint(graph->hashCode());
  if (footprintForward > 0) {
    if (__variableSpace->launchContext()->getWorkspace() != nullptr) {
      // this method will work only if current workspace size is smaller then proposed value
      sd_debug("Setting workspace to %lld bytes\n", footprintForward);
      __variableSpace->launchContext()->getWorkspace()->expandTo(footprintForward);
    }
  }

  // optionally saving graph build time
  if (Environment::getInstance().isProfiling()) flowPath->profile()->setBuildTime(GraphProfile::relativeTime(tb0));

  LongType timeStart = Environment::getInstance().isProfiling() ? GraphProfile::currentTime() : 0L;

  bool pe = graph->getExecutorConfiguration()->_executionMode == ::graph::ExecutionMode_AUTO;

  // basically if at some point code diverges, code branch might be _DISABLED_, and all nodes within that branch will be
  // disabled as well

  std::deque<LongType> frames;
  bool leftFrame = false;

  auto nodeTime = GraphProfile::currentTime();
  int lastId = -10000000;
  LongType exec_counter = 0;
  // we loop through op layers here
  for (int l = 0; l < (int)graph->getOnion()->size(); l++) {
    int layerSize = graph->getOnion()->count(l) == 1 ? graph->getOnion()->at(l)->size() : 0;

    int n = 0;
    // this omp block will probably never be the case
    for (; n < layerSize; n++) {
      if (++exec_counter > 10000) {
        l = graph->getOnion()->size();
        return Logger::logKernelFailureMsg("Early termination hit");
      }

      Node *node = graph->getOnion()->at(l)->at(n);

      if (Environment::getInstance().isProfiling()) flowPath->profile()->nodeById(node->id(), node->name()->c_str());

      if (lastId != node->id() && Environment::getInstance().isProfiling()) {
        if (lastId != -10000000)
          flowPath->profile()->nodeById(lastId)->setTotalTime(GraphProfile::relativeTime(nodeTime));

        lastId = node->id();
        nodeTime = GraphProfile::currentTime();
      }

      sd_debug("Step: %lld; Node: %i <%s>\n", exec_counter, node->id(), node->name()->c_str());

      // on first non-Exit node after loop we can rewind (if planned)
      if (!(node->opType() == ::graph::OpType_LOGIC && node->opNum() == logic::Exit)) {
        // VALIDATED

        // if we're out of frame - let's remove it from queue
        if (leftFrame) {
          auto frame_id = frames.back();
          frames.pop_back();
          flowPath->markFrameActive(frame_id, false);
          flowPath->forgetFrame(frame_id);

          leftFrame = false;
        }

        // TODO: move inactivity check right here
        bool shouldSkip = false;
        if (node->opType() == ::graph::OpType_LOGIC && node->opNum() == logic::Merge) {
          // Merge node has own checkout logic

          auto inputId0 = node->input()->at(0);
          auto inputId1 = node->input()->at(1);

          // Merge node can be skipped only both inputs are inactive
          if (!flowPath->isNodeActive(inputId0.first) && !flowPath->isNodeActive(inputId1.first)) shouldSkip = true;

        } else {
          // let's check for input nodes, if they are disabled or contain divergents
          for (size_t e = 0; e < node->input()->size(); e++) {
            auto inputId = node->input()->at(e);

            // not a node. skipping checks
            if (graph->getMapped()->count(inputId.first) == 0) continue;

            /**
             * We can skip current node, in two cases:
             * 1) If previous node was disabled
             * 2) If previous node was divergent node (i.e. IF op) and code went other way
             */
            Node *prevNode = graph->getMapped()->at(inputId.first);
            if (!flowPath->isNodeActive(inputId.first)) {
              shouldSkip = true;
              flowPath->markNodeActive(node->id(), false);

              sd_debug("Skipping Node_%i due to inactive input [%i]\n", node->id(), inputId.first);
              break;

            } else if (prevNode->isDivergencePoint()) {  // literally checking for switch here
              if (flowPath->branch(inputId.first) != inputId.second) {
                shouldSkip = true;
                flowPath->markNodeActive(node->id(), false);
                sd_debug("Skipping Node_%i due to divergent branch [%i]\n", node->id(), inputId.first);
                break;
              }
            }
          }
        }

        if (shouldSkip) continue;
      }

      // we're propagating frameId here (but only if wasn't set earlier)
      if (frames.size() > 0 && node->getFrameId() < 0) node->setFrameId(frames.back());

      flowPath->markNodeActive(node->id(), true);

      if (node->opType() == ::graph::OpType_LOGIC && node->opNum() == logic::Enter) {
        // Enter operation
        // VALIDATED

        // we expect this node to have frameId set
        auto frame_id = node->getFrameId();

        // new frame starts here
        if (frames.size() == 0 || (frames.size() > 0 && frames.back() != frame_id)) {
          flowPath->registerFrame(frame_id);
          frames.emplace_back(frame_id);
        }

        auto status = LogicExecutor::processNode(graph, node);
        if (status != Status::OK) return status;

      } else if (node->opType() == ::graph::OpType_LOGIC && node->opNum() == logic::NextIteration) {
        /**
         * NextIteration is special case: after successful execution of this op - we're changing execution position
         */
        // VALIDATED

        auto status = LogicExecutor::processNode(graph, node);
        if (status != Status::OK) return status;

        auto frame_id = frames.back();

        flowPath->markNodeActive(node->id(), true);
        flowPath->markExecuted(node->id(), true);

        if (!flowPath->isRewindPlanned(frame_id)) {
          auto nextLayer = node->getRewindLayer();

          sd_debug("Node_%i planned rewind to Node_%i at [%i:%i]\n", node->id(), node->getRewindNode(), nextLayer.first,
                   nextLayer.second);

          flowPath->planRewind(frame_id, true);
          flowPath->setRewindPositionOnce(frame_id, nextLayer.first - 1);

          continue;
        }

      } else if (node->opType() == ::graph::OpType_LOGIC && node->opNum() == logic::Exit) {
        // Exit node is another special case: it can rewind executioner to specific point in graph
        // VALIDATED

        auto frame_id = frames.back();

        // if this loop frame wasn't activated - just skip it
        if (!flowPath->isFrameActive(frame_id)) {
          flowPath->markNodeActive(node->id(), false);

          leftFrame = true;
          continue;
        }

        if (flowPath->isRewindPlanned(frame_id)) {
          // just break loop here
          l = flowPath->getRewindPosition(frame_id);
          flowPath->setRewindPosition(frame_id, -1);
          flowPath->planRewind(frame_id, false);

          break;
        } else {
          // execute Exit node otherwise

          auto status = LogicExecutor::processNode(graph, node);
          if (status != Status::OK) return status;

          leftFrame = true;
        }

      } else if (node->opType() == ::graph::OpType_LOGIC) {
        /**
         * If this LOGIC op, we'll use another execution model here
         */
        auto status = LogicExecutor::processNode(graph, node);

        if (status != Status::OK) return status;
      } else {
        auto timeStart2 = std::chrono::system_clock::now();

        // actual node execution happens right here
        Status status = executeFlatNode(graph, node, __variableSpace);

        auto timeEnd = std::chrono::system_clock::now();

        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart2).count();

        flowPath->setOuterTime(node->id(), outerTime);

        if (status != Status::OK) return status;

        // here we should handle divergent ops, and disable nodes accordingly
        if (node->isDivergencePoint()) {
          auto activeBranch = flowPath->branch(node->id());
          sd_debug("Active branch at node [%i]: %i\n", node->id(), activeBranch);

          // now we skip all branches except of this active one
        }

        if (Environment::getInstance().isDebugAndVerbose()) {
          if (__variableSpace->getVariable(node->id())->hasNDArray()) {
            auto array = __variableSpace->getVariable(node->id())->getNDArray();
            auto shape = ShapeUtils::shapeAsString(array);
            auto values = array->asIndexedString(16);
            auto type = DataTypeUtils::asString(array->dataType());
            sd_debug("node_%i finished. result shape: %s; data type: %s; first values: %s\n", node->id(), shape.c_str(),
                     type.c_str(), values->c_str());
          } else if (__variableSpace->getVariable(node->id())->hasNDArrayList()) {
            auto list = __variableSpace->getVariable(node->id())->hasNDArrayList()
                        ? __variableSpace->getVariable(node->id())->getNDArrayList()
                        : nullptr;
            sd_debug("node_% is ListOp, skipping evaluation", node->id());
          } else {
            sd_debug("node_% is Unknown: has no NDArray or NDArrayList", node->id());
          }
        }
      }

      // if node was executed - tag it as active
      flowPath->markExecuted(node->id(), true);
    }
  }

  // optionally saving execution time
  if (Environment::getInstance().isProfiling()) {
    flowPath->profile()->nodeById(lastId)->setTotalTime(GraphProfile::relativeTime(nodeTime));
    flowPath->profile()->setExecutionTime(GraphProfile::relativeTime(timeStart));
  }

  // saving memory footprint for current run
  if (__variableSpace->launchContext()->getWorkspace() != nullptr) {
    auto m = __variableSpace->launchContext()->getWorkspace()->getAllocatedSize();
    auto h = graph->hashCode();
    memory::MemoryRegistrator::getInstance().setGraphMemoryFootprintIfGreater(h, m);
  }

  if (tempFlow) {
    delete flowPath;
    __variableSpace->setFlowPath(nullptr);
  }

  return Status::OK;
}

/**
 * This method is provided for IPC:
 * 1) it accepts pointer to FlatBuffers buffer
 * 2) restores Graph from it
 * 3) Executes this Graph
 * 4) Packs execution results into FlatBuffers (FlatResults instance)
 * 5) Returns pointer to FlatBuffer results buffer
 *
 */
ResultWrapper *GraphExecutioner::executeFlatBuffer(Pointer pointer) {
  uint8_t *buffer = reinterpret_cast<uint8_t *>(pointer);


  auto restoredGraph = ::graph::GetFlatGraph(buffer);


  // converting FlatGraph to internal representation
  auto nativeGraph = new Graph(restoredGraph);

  if (Environment::getInstance().isDebugAndVerbose()) {
    nativeGraph->printOut();
  }

  FlowPath flowPath;
  nativeGraph->getVariableSpace()->setFlowPath(&flowPath);

  // sd_debug("Going to execute graph\n", 0);

  // executing internal representation
  auto status = execute(nativeGraph);
  if (status != Status::OK) {
    sd_printf("Graph execution failed with status: [%i]\n", status) return nullptr;
  }


  flatbuffers::FlatBufferBuilder builder(1024);

  // fetching time reports
  std::vector<flatbuffers::Offset<::graph::FlatTiming>> timings_vector;
  for (int e = 0; e < (int)nativeGraph->getAllNodes()->size(); e++) {
    Node *node = nativeGraph->getAllNodes()->at(e);

    if (node->getContextPrototype() == nullptr) continue;

    auto pair = ::graph::CreateLongPair(builder, flowPath.outerTime(node->id()), flowPath.innerTime(node->id()));
    if (node->getName() != nullptr) {
      auto name = builder.CreateString(node->getName()->c_str());
      auto fr = CreateFlatTiming(builder, node->id(), name, pair);
      timings_vector.push_back(fr);
    } else {
      auto fr = CreateFlatTiming(builder, node->id(), 0, pair);
      timings_vector.push_back(fr);
    }
  }

  // now, we'll prepare output, depending on given outputmode
  auto outputs = nativeGraph->fetchOutputs();
  auto size = static_cast<int>(outputs->size());
  int arrays = 0;
  std::vector<flatbuffers::Offset<::graph::FlatVariable>> variables_vector;
  for (int e = 0; e < size; e++) {
    auto var = outputs->at(e);

    // FIXME: we want to export multi-output nodes as well
    // FIXME: we want to export NDArrayList and skip nodes without outputs
    if (!var->hasNDArray()) continue;

    auto array = var->getNDArray();

    auto fArray = FlatUtils::toFlatArray(builder, *array);

    auto fName = builder.CreateString(*(var->getName()));
    auto id = ::graph::CreateIntPair(builder, var->id(), var->index());

    auto fv = CreateFlatVariable(builder, id, fName, static_cast<::graph::DType>(array->dataType()), 0, fArray);

    variables_vector.push_back(fv);
    arrays++;
  }

  sd_debug("Returning %i variables back\n", arrays);

  auto varTimings = builder.CreateVector(timings_vector);
  auto varVectors = builder.CreateVector(variables_vector);
  auto result = CreateFlatResult(builder, restoredGraph->id(), varVectors, varTimings);
  builder.Finish(result);

  // we might want to keep this graph for future
  delete outputs;
  delete nativeGraph;

  char *res = new char[builder.GetSize()];
  memcpy(res, builder.GetBufferPointer(), builder.GetSize());

  sd_debug("Buffer size: %lld\n", static_cast<sd::LongType>(builder.GetSize()));

  return new ResultWrapper(builder.GetSize(), reinterpret_cast<Pointer>(res));
}

Graph *GraphExecutioner::importFromTensorFlow(const char *fileName) {
  return nullptr;
}

/**
 *   This function returns file size for the given file name, or -1 if something went wrong
 */
long getFileSize(const char *filename) {
  struct stat stat_buf;
  int rc = stat(filename, &stat_buf);
  return rc == 0 ? stat_buf.st_size : -1;
}

/**
 *   Helper function, that loads given filename into uint8_t array
 *
 */
uint8_t *readFlatBuffers(const char *filename) {
  long fileLen = getFileSize(filename);
  if (fileLen < 0) {
    sd_printf("File [%s] wasn't found. Please check path and permissions\n", filename);
    THROW_EXCEPTION("File not found");
  }

  sd_debug("File length: %i\n", fileLen);

  uint8_t *data = new uint8_t[fileLen];

  FILE *in = fopen(filename, "rb");
  int cnt = 0;
  int b = 0;
  while (cnt < fileLen) {
    b = fread(data + cnt, 1, 1, in);

    cnt += b;
  }
  fclose(in);

  return data;
}

flatbuffers::Offset<::graph::FlatResult> GraphExecutioner::execute(Graph *graph, flatbuffers::FlatBufferBuilder &builder,
                                                          const ::graph::FlatInferenceRequest *request) {
  ExecutionResult result;
  auto varSpace = graph->getVariableSpace();

  if (request != nullptr && request->variables() != nullptr) {
    auto vars = request->variables();
    for (size_t e = 0; e < vars->size(); e++) {
      auto fv = vars->Get(e);
      auto v = new Variable(fv);
      varSpace->replaceVariable(v);
    }
  }

  if (Environment::getInstance().isDebugAndVerbose()) graph->printOut();

  auto status = execute(graph);
  if (status != Status::OK) throw graph_execution_exception(request->id());

  auto outputs = graph->fetchOutputs();

  if (outputs->size() == 0) throw no_results_exception(request->id());

  for (auto v : *outputs) {
    result.emplace_back(v);
  }

  auto t = result.asFlatResult(builder);

  delete outputs;

  return t;
}

/**
 *   This method reads given FlatBuffers file, and returns Graph instance
 *
 *   PLEASE NOTE: This method is mostly suited for tests and debugging/profiling
 */
Graph *GraphExecutioner::importFromFlatBuffers(const char *filename) {
  auto data = readFlatBuffers(filename);
  auto restoredGraph = importFromFlatPointer(reinterpret_cast<Pointer>(data));
   delete[] data;
  return restoredGraph;
}

Graph *GraphExecutioner::importFromFlatPointer(Pointer ptr) {
  auto fg = ::graph::GetFlatGraph(reinterpret_cast<uint8_t *>(ptr));
  auto restoredGraph = new Graph(fg);

  return restoredGraph;
}
}  // namespace graph
}  // namespace sd
