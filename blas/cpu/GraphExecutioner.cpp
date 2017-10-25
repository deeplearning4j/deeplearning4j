//
// @author raver119@gmail.com
//

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>
#include <graph/generated/result_generated.h>

//#include <protobuf/core/framework/graph.pb.h>

#include <Variable.h>
#include <VariableSpace.h>
#include <Node.h>
#include <Scope.h>
#include <GraphExecutioner.h>
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

    if (node->id() == 3)
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
        int cnt = 0;
        for (Variable<T>* v: *embedded->getPlaceholders()) {
            if (v->getName() != nullptr && v->getName()->size() > 0) {
                
                // trying symbolic lookup first
                if (variableSpace->hasVariable(v->getName())) {
                    // symbolic feeder
                    auto array = variableSpace->getVariable(v->getName())->getNDArray();
                    v->setNDArray(array->dup(array->ordering()));
                } else {
                    nd4j_debug("Can't find variable [%s] in parent graph...", v->getName()->c_str());
                    return ND4J_STATUS_BAD_INPUT;
                    //throw "Can't find desired variable";
                }
            } else {
                // if we're not using symbolic lookup - we'll use sequential approach then
                auto p = node->input()->at(cnt);
                auto array = variableSpace->getVariable(p)->getNDArray();
                v->setNDArray(array->dup(array->ordering()));
            }

            cnt++;
        }

        // executing embedded graph as independent one
        Nd4jStatus status = GraphExecutioner<T>::execute(embedded);
        if (status != ND4J_STATUS_OK)
            return status;

        //  now we should migrate its results to this node, as its own outputs
        cnt = 0;
        for (auto v: *embedded->fetchOutputs()){
            NDArray<T> *array = v->getNDArray();
            v->setNDArray(nullptr);

            if (cnt == 0)
                variableSpace->getVariable(node->id())->setNDArray(array);

            std::pair<int,int> pair(node->id(), cnt++);
            variableSpace->getVariable(pair)->setNDArray(array);
        }
        nd4j_debug("Embedded graph execution finished. %i variable(s) migrated\n", cnt);

    } else if (node->hasCustomOp()) {
        // if we have something to execute - lets just execute it.
        return node->getCustomOp()->execute(node->getBlock());
    }
  // This is legacy draft code
  // TODO: TO BE REMOVED
 /*   } else if (opType == OpType_TRANSFORM) {
        auto in = node->input()->at(0);

        auto x = variableSpace->getVariable(in);
        auto z = variableSpace->getVariable(node->id());

        // if node has only one input - that's regular TRANSFORM
        if (node->input()->size() == 1) {

            // if output of previous node is used in different code branches - duplicate it
            if (in.first > 0)
                if (graph->getMapped()->at(in.first)->output()->size() > 1) {
                    if (z->getNDArray() == nullptr) {
                        auto array = new NDArray<T>(x->getNDArray());
                        z->setNDArray(array);
                    }
                }


            // this assumes inplace operation
            if (z->getNDArray() == nullptr)
                z->setNDArray(x->getNDArray()->getView());


            functions::transform::Transform<T>::template exec(opNum, x->getNDArray()->getBuffer(),
                                                              x->getNDArray()->getShapeInfo(),
                                                              z->getNDArray()->getBuffer(),
                                                              z->getNDArray()->getShapeInfo(), node->extraParams(),
                    // FIXME: for some cases we NEED these vars
                                                              nullptr, nullptr);

        } else {
            // otherwise that's PAIRWISE op

            auto y = variableSpace->getVariable(node->input()->at(1));

            if (node->output()->size() > 0) {
                if (z->getNDArray() == nullptr) {
                    auto array = new NDArray<T>(x->getNDArray());
                    z->setNDArray(array);
                }
            }

            // this assumes inplace operation
            if (z->getNDArray() == nullptr)
                z->setNDArray(x->getNDArray()->getView());

            functions::pairwise_transforms::PairWiseTransform<T>::template exec(opNum, x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(), y->getNDArray()->getBuffer(), y->getNDArray()->getShapeInfo(),
                                                                                z->getNDArray()->getBuffer(), z->getNDArray()->getShapeInfo(), node->extraParams());
        }

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) >= 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }
    }  else if (opType == OpType_SCALAR) {
        auto in = node->input()->at(0);

        auto x = variableSpace->getVariable(in);

        // if output of previous node is used in different code branches - duplicate it
        auto z = variableSpace->getVariable(node->id());
        if (in.first > 0)
            if (graph->getMapped()->at(in.first)->output()->size() > 1) {
                auto array = new NDArray<T>(x->getNDArray());
                z->setNDArray(array);
            };

        // this assumes inplace op
        if (z->getNDArray() == nullptr)
            z->setNDArray(x->getNDArray()->getView());

        nd4j_verbose("xLength: %i\n", x->getNDArray()->lengthOf());
        nd4j_verbose("SCALAR BEFORE: X[0]: %f; X[1]: %f; scalar: %f\n", x->getNDArray()->getScalar(0), x->getNDArray()->getScalar(1), node->scalar());

        functions::scalar::ScalarTransform<T>::transform(opNum, x->getNDArray()->getBuffer(),
                                                              x->getNDArray()->getShapeInfo(),
                                                              z->getNDArray()->getBuffer(),
                                                              z->getNDArray()->getShapeInfo(),
                                                              node->scalar(),
                                                              node->extraParams());

        nd4j_verbose("SCALAR AFTER: Z[0]: %f; Z[1]: %f; scalar: %f\n", z->getNDArray()->getScalar(0), z->getNDArray()->getScalar(1), node->scalar());

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }
    }else if (opType == OpType_SUMMARYSTATS) {
        auto x = variableSpace->getVariable(node->input()->at(0));

        auto z = variableSpace->getVariable(node->id());
        // if there's no dimensions set - it's reduceToScalar
        if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
            if (z->getNDArray() == nullptr) {
                //z = new Variable<T>(new NDArray<T>(1, 1, 'c'));
                //z->setName(node->getName());
                z->setNDArray(new NDArray<T>(1,1,'c'));
            }
            z->getNDArray()->getBuffer()[0] = functions::summarystats::SummaryStatsReduce<T>::template execScalar(opNum, node->scalar() != 0.0, x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(), node->extraParams());

        } else {
            // dimensional reduction
            shape::TAD *tad = new shape::TAD(x->getNDArray()->getShapeInfo(), node->getDimensionsPtr(), node->getDimensions()->size());
            tad->createTadOnlyShapeInfo();
            tad->createOffsets();

            int resultLength = x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

            if (z->getNDArray() == nullptr || z->getNDArray()->lengthOf() != resultLength) {
                // FIXME: this is potentially bad
                if (z->getNDArray() != nullptr)
                    delete z->getNDArray();

                z->setNDArray(new NDArray<T>(1, resultLength, 'c'));
            }

            functions::summarystats::SummaryStatsReduce<T>::template exec(opNum, node->scalar() != 0.0, x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(), node->extraParams(), z->getNDArray()->getBuffer(), z->getNDArray()->getShapeInfo(),
                                                                    node->getDimensionsPtr() , node->getDimensions()->size());

            delete tad;
        }

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }
    } else if (opType == OpType_ACCUMULATION) {
        auto x = variableSpace->getVariable(node->input()->at(0));
        auto z = variableSpace->getVariable(node->id());

        //  regular accumulation with 1 argument
        if (node->input()->size() == 1) {
            // if there's no dimensions set - it's reduceToScalar
            if (node->getDimensions()->size() == 0 ||
                (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                nd4j_verbose("ACCUM SCALAR BEFORE: X[0]: %f; X[1]: %f; xLength: %i\n",
                             x->getNDArray()->getScalar(0), x->getNDArray()->getScalar(1),
                             x->getNDArray()->lengthOf());

                if (z->getNDArray() == nullptr) {
                    z->setNDArray(new NDArray<T>(1, 1, 'c'));
                }

                z->getNDArray()->getBuffer()[0] = functions::reduce::ReduceFunction<T>::template execScalar(opNum,
                                                                                                        x->getNDArray()->getBuffer(),
                                                                                                        x->getNDArray()->getShapeInfo(),
                                                                                                        node->extraParams());

                nd4j_verbose("ACCUM SCALAR  AFTER: Z[0]: %f; xLength: %i;\n", z->getNDArray()->getScalar(0),
                             x->getNDArray()->lengthOf());
            } else {
                // dimensional reduction
                shape::TAD *tad = new shape::TAD(x->getNDArray()->getShapeInfo(), node->getDimensionsPtr(),
                                                 node->getDimensions()->size());
                tad->createTadOnlyShapeInfo();
                tad->createOffsets();

                int resultLength =
                        x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

                if (z->getNDArray() == nullptr || z->getNDArray()->lengthOf() != resultLength) {
                    // FIXME: this is potentially BAD
                    if (z->getNDArray() != nullptr)
                        delete z->getNDArray();

                    z->setNDArray(new NDArray<T>(1, resultLength, 'c'));
                }

                functions::reduce::ReduceFunction<T>::template exec(opNum, x->getNDArray()->getBuffer(),
                                                                    x->getNDArray()->getShapeInfo(),
                                                                    node->extraParams(),
                                                                    z->getNDArray()->getBuffer(),
                                                                    z->getNDArray()->getShapeInfo(),
                                                                    node->getDimensionsPtr(),
                                                                    node->getDimensions()->size(),
                                                                    tad->tadOnlyShapeInfo, tad->tadOffsets);

                delete tad;
            }
        } else {
            // otherwise we're on reduce3, and expect 2 inputs

            auto y = variableSpace->getVariable(node->input()->at(1));

            // if there's no dimensions set - it's reduceToScalar
            if (node->getDimensions()->size() == 0 ||
                (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                nd4j_verbose("ACCUM3 SCALAR BEFORE: X[0]: %f; X[1]: %f; xLength: %f\n",
                             x->getNDArray()->getScalar(0), x->getNDArray()->getScalar(1),
                             x->getNDArray()->lengthOf());

                if (z->getNDArray() == nullptr) {
                    z->setNDArray(new NDArray<T>(1, 1, 'c'));
                }

                z->getNDArray()->getBuffer()[0] = functions::reduce3::Reduce3<T>::template execScalar(opNum,
                                                                                                        x->getNDArray()->getBuffer(),
                                                                                                        x->getNDArray()->getShapeInfo(),
                                                                                                        node->extraParams(),
                                                                                                        y->getNDArray()->getBuffer(),
                                                                                                        y->getNDArray()->getShapeInfo());

                nd4j_verbose("ACCUM3 SCALAR  AFTER: Z[0]: %f; xLength: %i;\n", z->getNDArray()->getScalar(0),
                             x->getNDArray()->lengthOf());
            } else {
                // dimensional reduction
                shape::TAD *tad = new shape::TAD(x->getNDArray()->getShapeInfo(), node->getDimensionsPtr(),
                                                 node->getDimensions()->size());
                tad->createTadOnlyShapeInfo();
                tad->createOffsets();

                int resultLength =
                        x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

                if (z->getNDArray() == nullptr || z->getNDArray()->lengthOf() != resultLength) {
                    if (z->getNDArray() != nullptr)
                        delete z->getNDArray();

                    z->setNDArray(new NDArray<T>(1, resultLength, 'c'));
                }

                functions::reduce3::Reduce3<T>::template exec(opNum, x->getNDArray()->getBuffer(),
                                                                    x->getNDArray()->getShapeInfo(),
                                                                    node->extraParams(),
                                                                    y->getNDArray()->getBuffer(),
                                                                    y->getNDArray()->getShapeInfo(),
                                                                    z->getNDArray()->getBuffer(),
                                                                    z->getNDArray()->getShapeInfo(),
                                                                    node->getDimensionsPtr(),
                                                                    node->getDimensions()->size(),
                                                                    tad->tadOnlyShapeInfo, tad->tadOffsets);

                delete tad;
            }
        }

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }
    } else if (opType == OpType_INDEX_ACCUMULATION) {

        auto x = variableSpace->getVariable(node->input()->at(0));

        auto z = x;
        // if there's no dimensions set - it's reduceToScalar
        if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
            z = new Variable<T>(new NDArray<T>(1,1, 'c'), node->getName()->c_str(), node->id());
            z->setName(node->getName());
            z->getNDArray()->getBuffer()[0] = (T) functions::indexreduce::IndexReduce<T>::template execScalar(opNum, x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(), node->extraParams());

        } else {
            // dimensional reduction
            shape::TAD *tad = new shape::TAD(x->getNDArray()->getShapeInfo(), node->getDimensionsPtr(), node->getDimensions()->size());
            tad->createTadOnlyShapeInfo();
            tad->createOffsets();

            int resultLength = x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

            z = new Variable<T>(new NDArray<T>(1, resultLength, 'c'), node->getName()->c_str(), node->id());
            z->setName(node->getName());
            functions::indexreduce::IndexReduce<T>::template exec(opNum, x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(), node->extraParams(), z->getNDArray()->getBuffer(), z->getNDArray()->getShapeInfo(),
                                                                    node->getDimensionsPtr() , node->getDimensions()->size(),
                                                                    tad->tadOnlyShapeInfo, tad->tadOffsets);

            delete tad;
        }

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }

    } else if (opType == OpType_BROADCAST) {
        auto x = variableSpace->getVariable(node->input()->at(0));
        auto y = variableSpace->getVariable(node->input()->at(1));

        auto z = x;

        auto tad = new shape::TAD(x->getNDArray()->getShapeInfo(), node->getDimensionsPtr(), node->getDimensions()->size());
        tad->createTadOnlyShapeInfo();
        tad->createOffsets();

        functions::broadcast::Broadcast<T>::exec(opNum,
                                                     x->getNDArray()->getBuffer(), x->getNDArray()->getShapeInfo(),
                                                     y->getNDArray()->getBuffer(), y->getNDArray()->getShapeInfo(),
                                                     z->getNDArray()->getBuffer(), z->getNDArray()->getShapeInfo(),
                                                     node->getDimensionsPtr(), node->getDimensions()->size(),
                                                     tad->tadOnlyShapeInfo, tad->tadOffsets,

                                                     // FIXME: this is bad. We have use case of different tads for broadcast
                                                     tad->tadOnlyShapeInfo, tad->tadOffsets);


        delete tad;

        variableSpace->putVariable(node->id(), z);

        if (node->hasExternalOutputs()) {
            for (unsigned int e = 0; e < node->output()->size(); e++) {
                if (node->output()->at(e) > 0)
                    continue;

                auto out = variableSpace->getVariable(node->output()->at(e));

                if (out->isEmpty()) {
                    out->setNDArray(z->getNDArray()->dup(z->getNDArray()->ordering()));
                } else {
                    // assign output
                    if (out->getNDArray() != z->getNDArray())
                        out->getNDArray()->assign(z->getNDArray());
                }
            }
        }
    }
*/
    return ND4J_STATUS_OK;
}


/**
 * This method executes given Graph instance, and returns error code.
 *
 * @param graph
 * @return one of error codes defined in pointercast.h
 */
template <typename T>
Nd4jStatus GraphExecutioner<T>::execute(Graph<T> *graph) {
    graph->buildGraph();
    auto __variableSpace = graph->getVariableSpace();

    bool pe = graph->getExecutorConfiguration()->_executionMode == ExecutionMode_AUTO;

    // TODO: add code divergence support here
    // basically if at some point code diverges, code branch might be _DISABLED_, and all nodes within that branch will be disabled as well

    // we loop through op layers here
    for (int l = 0; l < (int) graph->getOnion()->size(); l++) {
        int layerSize = graph->getOnion()->count(l) == 1 ? graph->getOnion()->at(l)->size() : 0;

//#pragma omp parallel for if (layerSize > 1 && pe) schedule(dynamic) proc_bind(spread)
        for (int n = 0; n < layerSize; n++) {
            Node<T>* node = graph->getOnion()->at(l)->at(n);

            /**
             * If this LOGIC op, we'll use another execution model here
             */
            if (node->opType() == OpType_LOGIC) {
                auto status = LogicExecutor<T>::processNode(graph, node);

                if (status == ND4J_STATUS_OK)
                    continue;
                else
                    return status;
            }

            bool shouldSkip = false;
            // let's check for input nodes, if they are disabled or contain divergents
            for (int e = 0; e < node->input()->size(); e++) {
                auto inputId = node->input()->at(e);

                // we're skipping external variables here
                if (inputId.first < 0)
                    continue;

                /**
                 * We can skip current node, in two cases:
                 * 1) If previous node was disabled
                 * 2) If previous node was divergent node (i.e. IF op) and code went other way
                 */
                Node<T>* prevNode = graph->getMapped()->at(inputId.first);
                if (!prevNode->isActive()) {
                    shouldSkip = true;
                    node->setActive(false);
                } else if (prevNode->isDivergencePoint()) {
                    if (prevNode->getBlock()->getBranch() != inputId.second) {
                        shouldSkip = true;
                        node->setActive(false);
                    }
                }
            }

            if (shouldSkip)
                continue;

            auto timeStart = std::chrono::system_clock::now();

            // actual node execution happens right here
            Nd4jStatus status = executeFlatNode(graph, node, __variableSpace);

            auto timeEnd = std::chrono::system_clock::now();

            auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

            if (node->getBlock() != nullptr)
                node->getBlock()->setOuterTime(outerTime);

            if (status != ND4J_STATUS_OK)
                return status;


            // here we should handle divergent ops, and disable nodes accordingly
            if (node->isDivergencePoint()) {
                auto activeBranch = node->getBlock()->getBranch();
                nd4j_debug("Active branch at node [%i]: %i\n", node->id(), activeBranch);

                // now we skip all branches except of this active one
            }

            if (nd4j::Environment::getInstance()->isDebugAndVerbose()) {
                NDArray<T> * array = __variableSpace->getVariable(node->id())->getNDArray();
                nd4j_debug("node_%i finished. result meanNumber: %f\n", node->id(), array->meanNumber());
            }
        }
    }

    return ND4J_STATUS_OK;
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
Nd4jPointer GraphExecutioner<T>::executeFlatBuffer(Nd4jPointer pointer) {
    uint8_t *buffer = reinterpret_cast<uint8_t *>(pointer);

    nd4j_debug("Trying to restore graph\n", 0);

    auto restoredGraph = GetFlatGraph(buffer);

    nd4j_debug("Graph restored\n", 0);

    // converting FlatGraph to internal representation
    auto nativeGraph = new Graph<T>(restoredGraph);

    nd4j_debug("Going to execute graph\n", 0);

    // executing internal representation
    GraphExecutioner<T>::execute(nativeGraph);

    nd4j_debug("Building output...\n", 0);

    flatbuffers::FlatBufferBuilder builder(1024);

    // fetching time reports
    std::vector<flatbuffers::Offset<FlatTiming>> timings_vector;
    for (int e = 0; e < (int) nativeGraph->getAllNodes()->size(); e++) {
        Node<T> *node = nativeGraph->getAllNodes()->at(e);

        if (node->getBlock() == nullptr)
            continue;

        auto pair = CreateLongPair(builder, node->getBlock()->getOuterTime(), node->getBlock()->getInnerTime());
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
    std::vector<flatbuffers::Offset<FlatVariable>> variables_vector;
    for (int e = 0; e < (int) outputs->size(); e++) {
        auto var = outputs->at(e);

        auto fShape = builder.CreateVector(var->getNDArray()->getShapeAsVector());
        auto fBuffer = builder.CreateVector(var->getNDArray()->getBufferAsVector());
        auto fName = builder.CreateString(*(var->getName()));

        auto fv = CreateFlatVariable(builder, var->id(), fName, fShape, fBuffer, -1);

        variables_vector.push_back(fv);
    }

    nd4j_debug("Returning %i variables back\n", variables_vector.size());

    auto varTimings = builder.CreateVector(timings_vector);
    auto varVectors = builder.CreateVector(variables_vector);
    auto result = CreateFlatResult(builder, restoredGraph->id(), varVectors, varTimings);
    builder.Finish(result);

    // we might want to keep this graph for future
    delete nativeGraph;

    return (Nd4jPointer) builder.GetBufferPointer();
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

                    Nd4jIndex __length = 0;

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
        throw "File not found";
    }

    nd4j_debug("File length: %i\n", fileLen);

    uint8_t * data = new uint8_t[fileLen];

    FILE *in = fopen(filename, "rb");
    int cnt = 0;

    while (cnt < fileLen) {
        fread(data + cnt, 1, 1, in);

        cnt++;
    }
    fclose(in);

    return data;
}


/**
*   This method reads given FlatBuffers file, and returns Graph instance
*
*   PLEASE NOTE: This method is mostly suited for tests and debugging/profiling
*/
template <typename T>
Graph<T>* GraphExecutioner<T>::importFromFlatBuffers(const char *filename) {
    uint8_t* data = readFlatBuffers(filename);

    auto fg = GetFlatGraph(data);
    auto restoredGraph = new Graph<float>(fg);

    delete[] data;

    return restoredGraph;
}


template class ND4J_EXPORT GraphExecutioner<float>;



}
}