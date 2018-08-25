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

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/result_generated.h>

//#include <protobuf/core/framework/graph.pb.h>

#include <Variable.h>
#include <VariableSpace.h>
#include <memory/MemoryRegistrator.h>
#include <Node.h>
#include <Scope.h>
#include <GraphExecutioner.h>
#include <graph/TimeHolder.h>
#include <loops/scalar.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>
#include <ops/declarable/DeclarableOp.h>

//#include <google/protobuf/text_format.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <fcntl.h>

#include <chrono>
#include <ctime>
#include <graph/execution/LogicExecutor.h>
#include <array/DataTypeUtils.h>
#include <helpers/BitwiseUtils.h>
#include <generated/array_generated.h>
#include <helpers/ShapeUtils.h>
#include <Status.h>
#include <deque>
#include <graph/ResultWrapper.h>
#include <graph/ExecutionResult.h>

namespace nd4j{
namespace graph {

/**
 * This method executes given Node (as in Op within Node)
 *
 * Basically it just does DeclarableOp::execute(Block<T>), and ops to their job. However, there are some additional functionality.
 *
 * @param graph - Graph instance pointer
 * @param node - Node instance pointer, which will be executed
 * @param variableSpace - VariableSpace instance pointer - varspace specific to current Thread/Session
 * @return
 */
template <typename T>
 Nd4jStatus GraphExecutioner<T>::executeFlatNode(Graph<T> *graph, Node<T> *node, VariableSpace<T> *variableSpace) {
    OpType opType = node->opType();
    int opNum = node->opNum();
//    std::string opName = *(node->getCustomOp()->getOpName());

    if (opType == OpType_BOOLEAN) {
        nd4j_debug("Executing boolean graph node_%i", node->id());
    } else if (opType == OpType_LOGIC) {
        nd4j_debug("Executing logic graph node_%i", node->id());
    } else if (opType == OpType_GRAPH) {
        nd4j_debug("Executing embedded graph node_%i", node->id());
    } else if (opType != OpType_CUSTOM) {
        nd4j_debug("Executing node_%i{%i}\n", node->id(), opNum);
    } else {
        nd4j_debug("Executing node_%i{%s}\n", node->id(), node->getCustomOp()->getOpName()->c_str());
    }

    Context<T> context(node->getContextPrototype(), variableSpace);

    if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
        //nd4j_debug("Input variables: %i\n", node->input()->size());
        printf("       Inputs: {");
        for (int e = 0; e < node->input()->size(); e++) {
            printf("[%i:%i]", node->input()->at(e).first, node->input()->at(e).second);

            if (e < node->input()->size() - 1)
                printf(", ");
        }
        printf("}\n");
        fflush(stdout);
    }

    if (node->id() == 13)
        nd4j_debug("","");

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
        //embedded->getExecutorConfiguration()->_outputMode = OutputMode_IMPLICIT;

        if (node->input()->size() != embedded->numberOfPlaceholders()) {
            nd4j_debug("Placeholders amount mismatch: %i expected, and %i available\n",node->input()->size(), embedded->numberOfPlaceholders());
            return ND4J_STATUS_BAD_INPUT;
        }

        // we need to propagate required variables to the embedded graph
        ResultSet<T> deletables;
        int cnt = 0;
        for (Variable<T>* v: *embedded->getPlaceholders()) {
            if (v->getName() != nullptr && v->getName()->size() > 0) {
                
                // trying symbolic lookup first
                if (variableSpace->hasVariable(v->getName())) {
                    // symbolic feeder
                    auto array = variableSpace->getVariable(v->getName())->getNDArray();
                    auto vr = array->dup();
//                    deletables.push_back(vr);
                    v->setNDArray(vr);
                } else {
                    nd4j_debug("Can't find variable [%s] in parent graph...", v->getName()->c_str());
                    return ND4J_STATUS_BAD_INPUT;
                    //throw "Can't find desired variable";
                }
            } else {
                // if we're not using symbolic lookup - we'll use sequential approach then
                auto p = node->input()->at(cnt);
                auto array = variableSpace->getVariable(p)->getNDArray();
                auto vr = array->dup();
                //deletables.push_back(vr);
                v->setNDArray(vr);
            }

            cnt++;
        }

        // executing embedded graph as independent one
        Nd4jStatus status = GraphExecutioner<T>::execute(embedded);
        if (status != ND4J_STATUS_OK)
            return status;

        //  now we should migrate its results to this node, as its own outputs
        cnt = 0;
        auto  outputs = embedded->fetchOutputs();

        for (auto v: *outputs){
            NDArray<T> *array = v->getNDArray();
            v->setNDArray(nullptr);

            std::pair<int,int> pair(node->id(), cnt++);

            auto var = variableSpace->getVariable(pair);

            //nd4j_printf("HasArray: [%i]; Removable: [%i]\n", var->hasNDArray(), var->isRemovable());
            var->setNDArray(array);
            var->markRemovable(true);
        }
        deletables.size();
        delete outputs;
        nd4j_debug("Embedded graph execution finished. %i variable(s) migrated\n", cnt);

    } else if (node->hasCustomOp()) {
        // if we have something to execute - lets just execute it.
        auto status = node->getCustomOp()->execute(&context);
        if (status != ND4J_STATUS_OK)
            return status;

        // propagate variables
        if (node->hasExternalOutputs()) {
            for (auto v: *node->output()) {
                if (variableSpace->hasExternalVariable(v.first)) {
                    variableSpace->getVariable(v.first)->getNDArray()->assign(variableSpace->getVariable(node->id())->getNDArray());
                }
            }
        }

        return status;
    }
    return ND4J_STATUS_OK;
}


/**
 * This method executes given Graph instance, and returns error code.
 *
 * @param graph
 * @return one of error codes defined in pointercast.h
 */
template <typename T>
Nd4jStatus GraphExecutioner<T>::execute(Graph<T> *graph, VariableSpace<T>* variableSpace) {
    auto __variableSpace = variableSpace == nullptr ? graph->getVariableSpace() : variableSpace;

    bool tempFlow = false;
    if (__variableSpace->flowPath() == nullptr) {
        tempFlow = true;
        __variableSpace->setFlowPath(new FlowPath());
    }
    auto flowPath = __variableSpace->flowPath();

    Nd4jLong tb0 = Environment::getInstance()->isProfiling() ? GraphProfile::currentTime() : 0L;
    graph->buildGraph();

    auto footprintForward = nd4j::memory::MemoryRegistrator::getInstance()->getGraphMemoryFootprint(graph->hashCode());
    if (footprintForward > 0) {
        if (__variableSpace->workspace() != nullptr) {
            // this method will work only if current workspace size is smaller then proposed value
            nd4j_debug("Setting workspace to %lld bytes\n", footprintForward);
            __variableSpace->workspace()->expandTo(footprintForward);
        }
    }

    // optionally saving graph build time
    if (Environment::getInstance()->isProfiling())
        flowPath->profile()->setBuildTime(GraphProfile::relativeTime(tb0));

    Nd4jLong timeStart = Environment::getInstance()->isProfiling() ? GraphProfile::currentTime() : 0L;

    bool pe = graph->getExecutorConfiguration()->_executionMode == ExecutionMode_AUTO;


    // basically if at some point code diverges, code branch might be _DISABLED_, and all nodes within that branch will be disabled as well

    std::deque<Nd4jLong> frames;
    bool inFrame =  false;
    bool leftFrame = false;

    auto nodeTime = GraphProfile::currentTime();
    int lastId = -10000000;
    Nd4jLong exec_counter = 0;
    // we loop through op layers here
    for (int l = 0; l < (int) graph->getOnion()->size(); l++) {
        int layerSize = graph->getOnion()->count(l) == 1 ? graph->getOnion()->at(l)->size() : 0;

        int n = 0;
// this omp block will probably never be the case
//#pragma omp parallel for if (layerSize > 1 && pe) schedule(dynamic) proc_bind(spread) private(n)
        for (; n < layerSize; n++) {
            if (++exec_counter > 10000) {
                l = graph->getOnion()->size();
                return Status::THROW("Early termination hit");
            }

            Node<T>* node = graph->getOnion()->at(l)->at(n);

            if (Environment::getInstance()->isProfiling())
                flowPath->profile()->nodeById(node->id(), node->name()->c_str());

            if (lastId != node->id() && Environment::getInstance()->isProfiling()) {
                if (lastId != -10000000)
                    flowPath->profile()->nodeById(lastId)->setTotalTime(GraphProfile::relativeTime(nodeTime));

                lastId = node->id();
                nodeTime = GraphProfile::currentTime();
            }

            nd4j_debug("Step: %lld; Node: %i <%s>\n", exec_counter, node->id(), node->name()->c_str());

            // on first non-Exit node after loop we can rewind (if planned)
            if (!(node->opType() == OpType_LOGIC && node->opNum() == 90L)) {
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
                if (node->opType() == OpType_LOGIC && node->opNum() == 60L) {
                    // Merge node has own checkout logic

                    auto inputId0 = node->input()->at(0);
                    auto inputId1 = node->input()->at(1);

                    // Merge node can be skipped only both inputs are inactive
                    if (!flowPath->isNodeActive(inputId0.first) && !flowPath->isNodeActive(inputId1.first))
                        shouldSkip = true;

                } else {
                    // let's check for input nodes, if they are disabled or contain divergents
                    for (int e = 0; e < node->input()->size(); e++) {
                        auto inputId = node->input()->at(e);

                        // we're skipping external variables here
                        if (inputId.first < 0 || __variableSpace->hasExternalVariable(inputId.first))
                            continue;

                        /**
                         * We can skip current node, in two cases:
                         * 1) If previous node was disabled
                         * 2) If previous node was divergent node (i.e. IF op) and code went other way
                         */
                        Node<T> *prevNode = graph->getMapped()->at(inputId.first);
                        if (!flowPath->isNodeActive(inputId.first)) {
                            shouldSkip = true;
                            flowPath->markNodeActive(node->id(), false);

                            nd4j_debug("Skipping Node_%i due to inactive input [%i]\n", node->id(), inputId.first);
                            break;

                        } else if (prevNode->isDivergencePoint()) { // literally checking for switch here
                            if (flowPath->branch(inputId.first) != inputId.second) {
                                shouldSkip = true;
                                flowPath->markNodeActive(node->id(), false);
                                nd4j_debug("Skipping Node_%i due to divergent branch [%i]\n", node->id(),
                                           inputId.first);
                                break;
                            }
                        }
                    }
                }

                if (shouldSkip)
                    continue;
            }

            // we're propagating frameId here (but only if wasn't set earlier)
            if (frames.size() > 0 && node->getFrameId() < 0)
                node->setFrameId(frames.back());


            flowPath->markNodeActive(node->id(), true);

            if (node->opType() == OpType_LOGIC && node->opNum() == 100L) {
                // Enter operation
                // VALIDATED

                // we expect this node to have frameId set
                auto frame_id = node->getFrameId();

                // new frame starts here
                if (frames.size() == 0 || (frames.size() > 0 && frames.back() != frame_id)) {
                    flowPath->registerFrame(frame_id);
                    frames.emplace_back(frame_id);
                    inFrame = true;
                }


                auto status = LogicExecutor<T>::processNode(graph, node);
                if (status != Status::OK())
                    return status;

            } else if (node->opType() == OpType_LOGIC && node->opNum() == 80L) {
                /**
                 * NextIteration is special case: after successful execution of this op - we're changing execution position
                 */
                // VALIDATED
                auto inputId = node->input()->at(0);

                auto status = LogicExecutor<T>::processNode(graph, node);
                if (status != Status::OK())
                    return status;

                auto frame_id = frames.back();

                flowPath->markNodeActive(node->id(), true);
                flowPath->markExecuted(node->id(), true);

                if (!flowPath->isRewindPlanned(frame_id)) {
                    auto nextLayer = node->getRewindLayer();

                    nd4j_debug("Node_%i planned rewind to Node_%i at [%i:%i]\n", node->id(), node->getRewindNode(), nextLayer.first, nextLayer.second);

                    flowPath->planRewind(frame_id, true);
                    flowPath->setRewindPositionOnce(frame_id, nextLayer.first - 1);

                    continue;
                }


            } else if (node->opType() == OpType_LOGIC && node->opNum() == 90L) {
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

                    auto status = LogicExecutor<T>::processNode(graph, node);
                    if (status != Status::OK())
                        return status;

                    leftFrame = true;
                }


            } else if (node->opType() == OpType_LOGIC) {
                /**
                 * If this LOGIC op, we'll use another execution model here
                 */
                auto status = LogicExecutor<T>::processNode(graph, node);

                if (status != Status::OK())
                    return status;
            } else {


                auto timeStart = std::chrono::system_clock::now();

                // actual node execution happens right here
                Nd4jStatus status = executeFlatNode(graph, node, __variableSpace);

                auto timeEnd = std::chrono::system_clock::now();

                auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();


                flowPath->setOuterTime(node->id(), outerTime);

                if (status != ND4J_STATUS_OK)
                    return status;


                // here we should handle divergent ops, and disable nodes accordingly
                if (node->isDivergencePoint()) {
                    auto activeBranch = flowPath->branch(node->id());
                    nd4j_debug("Active branch at node [%i]: %i\n", node->id(), activeBranch);

                    // now we skip all branches except of this active one
                }

                if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {

                    if (__variableSpace->getVariable(node->id())->hasNDArray()) {
                        auto array = __variableSpace->getVariable(node->id())->getNDArray();
                        auto shape = ShapeUtils<T>::shapeAsString(array);
                        auto values = array->asIndexedString(16);
                        nd4j_debug("node_%i finished. result shape: %s; first values: %s\n", node->id(), shape.c_str(), values.c_str());
                    } else if (__variableSpace->getVariable(node->id())->hasNDArrayList()) {
                        auto list = __variableSpace->getVariable(node->id())->hasNDArrayList() ? __variableSpace->getVariable(node->id())->getNDArrayList() : nullptr;
                        nd4j_debug("node_% is ListOp, skipping evaluation", node->id());
                    } else {
                        nd4j_debug("node_% is Unknown: has no NDArray or NDArrayList", node->id());
                    }
                }
            }

            // if node was executed - tag it as active
            flowPath->markExecuted(node->id(), true);
        }
    }

    // optionally saving execution time
    if (Environment::getInstance()->isProfiling()) {
        flowPath->profile()->nodeById(lastId)->setTotalTime(GraphProfile::relativeTime(nodeTime));
        flowPath->profile()->setExecutionTime(GraphProfile::relativeTime(timeStart));
        //flowPath->profile().printOut();
    }

    // saving memory footprint for current run
    if (__variableSpace->workspace() != nullptr) {
        auto m = __variableSpace->workspace()->getAllocatedSize();
        auto h = graph->hashCode();
        nd4j::memory::MemoryRegistrator::getInstance()->setGraphMemoryFootprintIfGreater(h, m);
    }

    if (tempFlow)
        delete flowPath;

    return Status::OK();
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
template <typename T>
    nd4j::graph::ResultWrapper* GraphExecutioner<T>::executeFlatBuffer(Nd4jPointer pointer) {
    uint8_t *buffer = reinterpret_cast<uint8_t *>(pointer);

    // nd4j_debug("Trying to restore graph\n", 0);

    auto restoredGraph = GetFlatGraph(buffer);

    // nd4j_debug("Graph restored\n", 0);

    // converting FlatGraph to internal representation
    auto nativeGraph = new Graph<T>(restoredGraph);

    if (Environment::getInstance()->isDebugAndVerbose()) {
        nativeGraph->printOut();
    }

    FlowPath flowPath;
    nativeGraph->getVariableSpace()->setFlowPath(&flowPath);


    // nd4j_debug("Going to execute graph\n", 0);

    // executing internal representation
    auto status = GraphExecutioner<T>::execute(nativeGraph);
    if (status != ND4J_STATUS_OK) {
        nd4j_printf("Graph execution failed with status: [%i]\n", status)
        return nullptr;
    }

    // nd4j_debug("Building output...\n", 0);

    flatbuffers::FlatBufferBuilder builder(1024);

    // fetching time reports
    std::vector<flatbuffers::Offset<FlatTiming>> timings_vector;
    for (int e = 0; e < (int) nativeGraph->getAllNodes()->size(); e++) {
        Node<T> *node = nativeGraph->getAllNodes()->at(e);

        if (node->getContextPrototype() == nullptr)
            continue;

        auto pair = CreateLongPair(builder, flowPath.outerTime(node->id()), flowPath.innerTime(node->id()));
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
    std::vector<flatbuffers::Offset<FlatVariable>> variables_vector;
    for (int e = 0; e < size; e++) {
        auto var = outputs->at(e);

        // FIXME: we want to export multi-output nodes as well
        // FIXME: we want to export NDArrayList and skip nodes without outputs
        if (!var->hasNDArray())
            continue;


        NDArray<T>* array = var->getNDArray();
        auto byteVector = array->asByteVector();

        auto fBuffer = builder.CreateVector(byteVector);
        auto fShape = builder.CreateVector(array->getShapeInfoAsFlatVector());

        auto bo = static_cast<nd4j::graph::ByteOrder>(BitwiseUtils::asByteOrder());

        auto fArray = CreateFlatArray(builder, fShape, fBuffer, static_cast<nd4j::graph::DataType>(DataTypeUtils::fromT<T>()), bo);

        auto fName = builder.CreateString(*(var->getName()));
        auto id = CreateIntPair(builder, var->id(), var->index());

        auto fv = CreateFlatVariable(builder, id, fName, 0, fArray);

        variables_vector.push_back(fv);
        arrays++;
    }

    nd4j_debug("Returning %i variables back\n", arrays);

    auto varTimings = builder.CreateVector(timings_vector);
    auto varVectors = builder.CreateVector(variables_vector);
    auto result = CreateFlatResult(builder, restoredGraph->id(), varVectors, varTimings);
    builder.Finish(result);

    // we might want to keep this graph for future
    delete outputs;
    delete nativeGraph;

    char* res = new char[builder.GetSize()];
    memcpy(res, builder.GetBufferPointer(), builder.GetSize());

    nd4j_debug("Buffer size: %lld\n", static_cast<Nd4jLong>(builder.GetSize()));

    return new ResultWrapper(builder.GetSize(), reinterpret_cast<Nd4jPointer>(res));
}


template <typename T>
Graph<T>* GraphExecutioner<T>::importFromTensorFlow(const char *fileName) {
    /*
    if (fileName == nullptr)
        return nullptr;

    int fd = open(fileName, O_RDONLY);

    if (fd < 0) {
        nd4j_printf("File not found: [%s]\n", fileName);
        return nullptr;
    }

    nd4j_verbose("Trying to load TF GraphDef from file [%s]\n", fileName);

    tensorflow::GraphDef graphDef;
    bool res = graphDef.ParseFromFileDescriptor(fd);

    // trying to read graph as text
    if(!res) {
        close(fd);
        fd = open(fileName, O_RDONLY);

        google::protobuf::io::FileInputStream fileInput(fd);
        fileInput.SetCloseOnDelete(true);

        if (!google::protobuf::TextFormat::Parse(&fileInput, &graphDef)) {
            nd4j_printf("Failed to read file\n","");
        } else {
            res = true;
        }
    }

    close(fd);

    if (!res)
        return nullptr;

    auto graph = new Graph<T>();
    auto variableSpace = graph->getVariableSpace();

    std::map<const std::string, int> variablesMap;

    int variablesCounter = 0;
    int nodesCounter = 0;
    nd4j_verbose("Number of nodes in graphDef: %i\n", graphDef.node_size());
    for (int n = 0; n < graphDef.node_size(); n++) {
        auto node = graphDef.node(n);

        // if that's external variable - we put it to variable space
        if (strcmp(TF_VAR, node.op().c_str()) == 0 || strcmp(TF_CONST, node.op().c_str()) == 0 || strcmp(TF_INPUT, node.op().c_str()) == 0) {
            nd4j_printf("Variable found: %s\n", node.name().c_str());
            auto variable = new Variable<T>();
            variable->setName(new std::string(node.name().c_str()));
            variable->setId(--variablesCounter);
            variableSpace->putVariable(variable->id(), variable);

            std::pair<const std::string, int> pair(node.name(), variable->id());
            variablesMap.insert(pair);

            // TODO: we might want to have something like that.
            // it basically just gives input validation option, since settles expectations for input
            if (strcmp(TF_INPUT, node.op().c_str()) == 0)
                continue;

            // checking shape, not applicable to input, since it can vary
            if (node.attr().count("shape")) {
                auto attr = node.attr().at("shape");
                int dims = attr.shape().dim_size();

                if (dims > 0) {
                    std::vector<int> __shape;

                    // we don't have rank1 arrays. vector is 2d.
                    if (dims == 1)
                        __shape.push_back(1);

                    // roll through dimensions
                    for (auto s: attr.shape().dim()) {
                        __shape.push_back((int) s.size()) ;
                    }

                    variable->setNDArray(new NDArray<T>('c', __shape));

                    nd4j_printf("Shape found: %i dims;\n", dims);
                    variable->getNDArray()->printShapeInfo();
                }
            }

            // checking tensor attached
            if (node.attr().count("value")) {
                auto attr = node.attr().at("value");

                // int
                if (attr.tensor().dtype() == ::tensorflow::DataType::DT_INT32) {
                    nd4j_verbose("Int size: %i\n", attr.tensor().int_val_size());

                    Nd4jLong __length = 0;

                    nd4j_verbose("Tensor has shape: %i\n", attr.tensor().has_tensor_shape());
                    if (attr.tensor().has_tensor_shape()) {
                        auto shape = attr.tensor().tensor_shape();
                        int dims = shape.dim_size();

                        if (dims > 0) {
                            std::vector<int> __shape;
                            // we don't have rank1 arrays. vector is 2d.
                            if (dims == 1)
                                __shape.push_back(1);

                            // roll through dimensions
                            for (auto s: shape.dim()) {
                                __shape.push_back((int) s.size());
                            }

                            variable->setNDArray(new NDArray<T>('c', __shape));
                            __length = variable->getNDArray()->lengthOf();

                            nd4j_printf("Tensor shape found: %i dims;\n", dims);
                            variable->getNDArray()->printShapeInfo();
                        }
                    }

                    // it can be valueOf array
                    if (attr.tensor().int_val_size() == 1 && __length > 0) {
                        variable->getNDArray()->assign((T) attr.tensor().int_val(0));
                    }
                }
            }
        } else {
            nd4j_verbose("Node id: [%i]; name: [%s]; opName: [%s]\n", n + 1, node.name().c_str(),
                         node.op().c_str());

            nd4j::ops::DeclarableOp<T> *op = nd4j::ops::OpRegistrator::getInstance()->getOperationFloat(node.op().c_str());

            if (op == nullptr) {
                nd4j_verbose("Op wasn't found: %s\n", node.op().c_str());
                return nullptr;
            }

            auto jNode = new Node<T>();
            jNode->setName(node.name());
            jNode->setId(++nodesCounter);
            jNode->setCustomOp(op);
            jNode->setBlock(new Block<T>(jNode->id(), variableSpace));

            std::pair<const std::string, int> pair(node.name(), jNode->id());
            variablesMap.insert(pair);

            // multi-output nodes require special treatment
            for (int e = 0; e < op->getOpDescriptor()->getNumberOfOutputs(); e++) {
                std::string deepName(node.name());
                deepName += ":" + std::to_string(e);
                auto deepVar = new Variable<T>();
                deepVar->setName(&deepName);

                if (e > 0)
                    deepVar->setId(--variablesCounter);
                else
                    deepVar->setId(jNode->id());

                std::pair<const std::string, int> pair(deepName, deepVar->id());
                variablesMap.insert(pair);

                variableSpace->putVariable(deepVar->id(), deepVar);

                std::pair<int, int> nodepair(jNode->id(), e);
                variableSpace->putVariable(nodepair, deepVar);
            }


            printf("             Inputs: [");
            for (int i = 0; i < node.input_size(); i++) {
                nd4j_printf("Trying input: %s\n", node.input(i).c_str());

                // if this fails - we're probably on partial input :)
                if (!variablesMap.count(node.input(i)))
                    return nullptr;

                printf("%s (%i)", node.input(i).c_str(), variablesMap.at(node.input(i)));


                jNode->pickInput(variablesMap.at(node.input(i)));
                jNode->getBlock()->pickInput(variablesMap.at(node.input(i)));


                if (i < node.input_size() + 1)
                    printf(", ");
            }
            printf("]\n");

            graph->addNode(jNode);
        }
    }

    return graph;
     */
    return nullptr;
}

/**
*   This function returns file size for the given file name, or -1 if something went wrong
*/
long getFileSize(const char * filename) {
    struct stat stat_buf;
    int rc = stat(filename, &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

/**
*   Helper function, that loads given filename into uint8_t array
*
*/
uint8_t* readFlatBuffers(const char * filename) {
    long fileLen = getFileSize(filename);
    if (fileLen < 0) {
        nd4j_printf("File [%s] wasn't found. Please check path and permissions\n", filename);
        throw std::runtime_error("File not found");
    }

    nd4j_debug("File length: %i\n", fileLen);

    uint8_t * data = new uint8_t[fileLen];

    FILE *in = fopen(filename, "rb");
    int cnt = 0;
    int b = 0;
    while (cnt < fileLen) {
        b += fread(data + cnt, 1, 1, in);

        cnt++;
    }
    fclose(in);

    return data;
}


template <typename T>
flatbuffers::Offset<FlatResult> GraphExecutioner<T>::execute(Graph<T> *graph, flatbuffers::FlatBufferBuilder &builder, FlatInferenceRequest* request) {
    ExecutionResult<T> result;

    return result.asFlatResult(builder);
}


/**
*   This method reads given FlatBuffers file, and returns Graph instance
*
*   PLEASE NOTE: This method is mostly suited for tests and debugging/profiling
*/
template <typename T>
Graph<T>* GraphExecutioner<T>::importFromFlatBuffers(const char *filename) {
    auto data = readFlatBuffers(filename);

    auto restoredGraph = importFromFlatPointer(reinterpret_cast<Nd4jPointer>(data));

    delete[] data;
    
    return restoredGraph;
}

    template <typename T>
    Graph<T> *GraphExecutioner<T>::importFromFlatPointer(Nd4jPointer ptr) {
        auto fg = GetFlatGraph(reinterpret_cast<uint8_t *>(ptr));
        auto restoredGraph = new Graph<T>(fg);

        return restoredGraph;
    }


        template class ND4J_EXPORT GraphExecutioner<float>;
        template class ND4J_EXPORT GraphExecutioner<float16>;
        template class ND4J_EXPORT GraphExecutioner<double>;
    }
}