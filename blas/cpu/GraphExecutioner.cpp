//
// @author raver119@gmail.com
//

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <Variable.h>
#include <VariableSpace.h>
#include <Node.h>
#include <GraphExecutioner.h>
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

            if (opType == OpType_TRANSFORM) {
                int in = node->input()->at(0);

                auto x = variableSpace->getVariable(in);

                // if output of previous node is used in different code branches - duplicate it
                if (in > 0)
                    if (graph->getMapped()->at(in)->output()->size() > 1) {
                        auto array = x->getNDArray()->dup(x->getNDArray()->ordering());
                        x = new Variable<float>(array);
                    };

                functions::transform::Transform<float>::template exec(opNum, x->getNDArray()->_buffer,
                                                                      x->getNDArray()->_shapeInfo,
                                                                      x->getNDArray()->_buffer,
                                                                      x->getNDArray()->_shapeInfo, node->extraParams(), nullptr,
                                                                      nullptr);

                variableSpace->putVariable(node->id(), x);

                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
                        if (node->output()->at(e) > 0)
                            continue;

                        auto out = variableSpace->getVariable(node->output()->at(e));

                        // assign output
                        if (out->getNDArray() != x->getNDArray())
                            out->getNDArray()->assign(x->getNDArray());
                    }
                }
            } else if (opType == OpType_PAIRWISE) {

                //printf("PWT> x: %i; y: %i\n", node->input()->at(0), node->input()->at(1));
                //fflush(stdout);

                auto x = variableSpace->getVariable(node->input()->at(0));
                auto y = variableSpace->getVariable(node->input()->at(1));

                //printf("PWT> X: %f; Y: %f\n", x->getNDArray()->getScalar(0), y->getNDArray()->getScalar(0));
                //fflush(stdout);

                auto z = x;
                if (node->output()->size() > 0) {
                    z = new Variable<float>(new NDArray<float>(x->getNDArray()));
                }


                functions::pairwise_transforms::PairWiseTransform<float>:: template exec(opNum, x->getNDArray()->_buffer, x->getNDArray()->_shapeInfo, y->getNDArray()->_buffer, y->getNDArray()->_shapeInfo,
                                                                                         z->getNDArray()->_buffer, z->getNDArray()->_shapeInfo, node->extraParams());

                variableSpace->putVariable(node->id(), z);


                if (node->hasExternalOutputs()) {
                    for (int e = 0; e < node->output()->size(); e++) {
                        if (node->output()->at(e) > 0)
                            continue;

                        auto out = variableSpace->getVariable(node->output()->at(e));

                        // assign output
                        if (out->getNDArray() != z->getNDArray())
                            out->getNDArray()->assign(z->getNDArray());
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

#pragma omp parallel for if (layerSize > 1) schedule(dynamic) proc_bind(spread)
                for (int n = 0; n < layerSize; n++) {
                    auto node = graph->getOnion()->at(l)->at(n);

                    executeFlatNode(graph, node, __variableSpace);
                }
            }

            return ND4J_STATUS_OK;
        }
    }
}