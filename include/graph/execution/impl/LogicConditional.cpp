//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicConditional.h>
#include <GraphExecutioner.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicConditional<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            int scopeConditionIndex = node->input()->at(0).first;
            int scopeFalseIndex = node->input()->at(1).first;
            int scopeTrueIndex = node->input()->at(2).first;

            auto scopeCondition = graph->scopeById(scopeConditionIndex);
            int lastNode = 0;
            for (auto v: *scopeCondition->nodes()) {
                GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                lastNode = v->id();
            }

            // now we should take result of the Scope run, and evaluate it
            //nd4j_debug("", "");
            auto result = __variableSpace->getVariable(lastNode)->getNDArray();
            //result->printBuffer("Result of the last node:");

            // now we're executing one of the scopes, depending on condition evaluation
            if (result->getScalar(0) == (T) 0.0f) {
                auto scopeFalse = graph->scopeById(scopeFalseIndex);
                lastNode = 0;
                for (auto v: *scopeFalse->nodes()) {
                    GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }
            } else {
                auto scopeTrue = graph->scopeById(scopeTrueIndex);
                lastNode = 0;
                for (auto v: *scopeTrue->nodes()) {
                    GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }
            }

            // now fetch and transfer variables to Conditional node
            for (int e = 0; e < 65536; e++) {
                std::pair<int, int> pair(lastNode, e);
                std::pair<int, int> pairNew(node->id(), e);
                if (__variableSpace->hasVariable(pair)) {
                    auto array = __variableSpace->getVariable(pair)->getNDArray();
                    auto newVar = new Variable<T>(array);
                    newVar->setId(lastNode, e);
                    newVar->markRemovable(false);

                    __variableSpace->putVariable(pairNew, newVar);
                } else
                    break;
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicConditional<float>;
//        template class ND4J_EXPORT LogicConditional<float16>;
//        template class ND4J_EXPORT LogicConditional<double>;
    }
}