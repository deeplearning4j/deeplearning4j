//
// @author raver119@gmail.com
//

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <Variable.h>
#include <VariableSpace.h>
#include <Node.h>
#include <GraphExecutioner.h>
#include <loops/scalar.h>
#include <loops/pairwise_transform.h>
#include <loops/transform.h>

namespace nd4j{
    namespace graph {

        /**
         *
         * @param graph
         * @param node
         * @param variableSpace
         * @return
         */
        static Nd4jStatus executeFlatNode(nd4j::graph::Graph *graph, nd4j::graph::Node *node, nd4j::graph::VariableSpace<float> *variableSpace) {
            OpType opType = node->opType();
            int opNum = node->opNum();

            printf("Executing node_%i{%i}\n", node->id(), opNum);
            fflush(stdout);

            if (opType == OpType_TRANSFORM) {
                int in = node->input()->at(0);

                auto x = variableSpace->getVariable(in);

                // if output of previous node is used in different code branches - duplicate it
                auto z = x;
                if (in > 0)
                    if (graph->getMapped()->at(in)->output()->size() > 1) {
                        auto array = new NDArray<float>(x->getNDArray());
                        z = new Variable<float>(array);
                    };

                functions::transform::Transform<float>::template exec(opNum, x->getNDArray()->_buffer,
                                                                      x->getNDArray()->_shapeInfo,
                                                                      z->getNDArray()->_buffer,
                                                                      z->getNDArray()->_shapeInfo, node->extraParams(),
                                                                      // FIXME: for some cases we NEED these vars
                                                                      nullptr, nullptr);

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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
            } else if (opType == OpType_PAIRWISE) {
                auto x = variableSpace->getVariable(node->input()->at(0));
                auto y = variableSpace->getVariable(node->input()->at(1));

                auto z = x;
                if (node->output()->size() > 0) {
                    z = new Variable<float>(new NDArray<float>(x->getNDArray()));
                }


                functions::pairwise_transforms::PairWiseTransform<float>::template exec(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, y->getNDArray()->_buffer, y->getNDArray()->_shapeInfo,
                                                                                         z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo, node->extraParams());

                variableSpace->putVariable(node->id(), z);


                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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
            } else if (opType == OpType_SCALAR) {
                int in = node->input()->at(0);

                auto x = variableSpace->getVariable(in);

                // if output of previous node is used in different code branches - duplicate it
                auto z = x;
                if (in > 0)
                    if (graph->getMapped()->at(in)->output()->size() > 1) {
                        auto array = new NDArray<float>(x->getNDArray());
                        z = new Variable<float>(array);
                    };

                functions::scalar::ScalarTransform<float>::transform(opNum, x->getNDArray()->_buffer,
                                                                      x->getNDArray()->_shapeInfo,
                                                                      z->getNDArray()->_buffer,
                                                                      z->getNDArray()->_shapeInfo,
                                                                      node->scalar(),
                                                                      node->extraParams());

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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

                auto z = x;
                // if there's no dimensions set - it's reduceToScalar
                if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                    z = new Variable<float>(new NDArray<float>(1,1, 'c'));
                    z->getNDArray()->_buffer[0] = functions::summarystats::SummaryStatsReduce<float>::template execScalar(opNum, node->scalar() != 0.0, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams());

                } else {
                    // dimensional reduction
                    shape::TAD *tad = new shape::TAD(x->getNDArray()->_shapeInfo, node->getDimensionsPtr(), node->getDimensions()->size());
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    int resultLength = x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

                    z = new Variable<float>(new NDArray<float>(1, resultLength, 'c'));


                    functions::summarystats::SummaryStatsReduce<float>::template exec(opNum, node->scalar() != 0.0, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams(), z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo,
                                                                            node->getDimensionsPtr() , node->getDimensions()->size());

                    delete tad;
                }

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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

                auto z = x;
                // if there's no dimensions set - it's reduceToScalar
                if (node->getDimensions()->size() == 0 || (node->getDimensions()->size() == 1 && node->getDimensions()->at(0) == MAX_INT)) {
                    z = new Variable<float>(new NDArray<float>(1,1, 'c'));
                    z->getNDArray()->_buffer[0] = functions::reduce::ReduceFunction<float>::template execScalar(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams());

                } else {
                    // dimensional reduction
                    shape::TAD *tad = new shape::TAD(x->getNDArray()->_shapeInfo, node->getDimensionsPtr(), node->getDimensions()->size());
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    int resultLength = x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

                    z = new Variable<float>(new NDArray<float>(1, resultLength, 'c'));

                    functions::reduce::ReduceFunction<float>::template exec(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams(), z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo,
                                                                            node->getDimensionsPtr() , node->getDimensions()->size(),
                                                                            tad->tadOnlyShapeInfo, tad->tadOffsets);

                    delete tad;
                }

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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
                    z = new Variable<float>(new NDArray<float>(1,1, 'c'));
                    z->getNDArray()->_buffer[0] = (float) functions::indexreduce::IndexReduce<float>::template execScalar(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams());

                } else {
                    // dimensional reduction
                    shape::TAD *tad = new shape::TAD(x->getNDArray()->_shapeInfo, node->getDimensionsPtr(), node->getDimensions()->size());
                    tad->createTadOnlyShapeInfo();
                    tad->createOffsets();

                    int resultLength = x->getNDArray()->lengthOf() / shape::length(tad->shapeInfoOnlyShapeAndStride());

                    z = new Variable<float>(new NDArray<float>(1, resultLength, 'c'));

                    functions::indexreduce::IndexReduce<float>::template exec(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, node->extraParams(), z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo,
                                                                            node->getDimensionsPtr() , node->getDimensions()->size(),
                                                                            tad->tadOnlyShapeInfo, tad->tadOffsets);

                    delete tad;
                }

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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

                auto tad = new shape::TAD(x->getNDArray()->_shapeInfo, node->getDimensionsPtr(), node->getDimensions()->size());
                tad->createTadOnlyShapeInfo();
                tad->createOffsets();

                functions::broadcast::Broadcast<float>::exec(opNum,
                                                             x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo,
                                                             y->getNDArray()->_buffer, y->getNDArray()->_shapeInfo,
                                                             z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo,
                                                             node->getDimensionsPtr(), node->getDimensions()->size(),
                                                             tad->tadOnlyShapeInfo, tad->tadOffsets,

                                                             // FIXME: this is bad. We have use case of different tads for broadcast
                                                             tad->tadOnlyShapeInfo, tad->tadOffsets);


                delete tad;

                variableSpace->putVariable(node->id(), z);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
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

            return ND4J_STATUS_OK;
        }


        /**
         *
         * @param graph
         * @return
         */
        Nd4jStatus nd4j::graph::GraphExecutioner::execute(nd4j::graph::Graph *graph) {
            auto __variableSpace = graph->getVariableSpace();

            // we loop through op layers here
            for (int l = 0; l < graph->getOnion()->size(); l++) {
                int layerSize = graph->getOnion()->count(l) == 1 ? graph->getOnion()->at(l)->size() : 0;

//#pragma omp parallel for if (layerSize > 1) schedule(dynamic) proc_bind(spread)
                for (int n = 0; n < layerSize; n++) {
                    auto node = graph->getOnion()->at(l)->at(n);

                    executeFlatNode(graph, node, __variableSpace);
                }
            }

            return ND4J_STATUS_OK;
        }
    }
}