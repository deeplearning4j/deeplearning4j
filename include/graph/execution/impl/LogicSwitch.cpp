//
// Created by raver119 on 21.10.17.
//

#include <pointercast.h>
#include <graph/execution/LogicSwitch.h>
#include <GraphExecutioner.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicSwitch<T>::processNode(Graph<T>* graph, Node<T>* node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();
            Context<T> ctx(node->getContextPrototype(), __variableSpace);

            int scopeConditionIndex = node->input()->at(0).first;
            auto input = ctx.variable(1);

            auto scopeCondition = graph->scopeById(scopeConditionIndex);
            int lastNode = 0;
            for (auto v: *scopeCondition->nodes()) {
                GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                lastNode = v->id();
            }

            // now we should take result of the Scope run, and evaluate it
            auto result = __variableSpace->getVariable(lastNode)->getNDArray();
            //result->printBuffer("Result of the last node");


            std::pair<int, int> pair0(node->id(), 0);
            std::pair<int, int> pair1(node->id(), 1);

            if (!__variableSpace->hasVariable(pair0))
                __variableSpace->putVariable(pair0, new Variable<T>(nullptr, nullptr, node->id(), 0));

            if (!__variableSpace->hasVariable(pair1))
                __variableSpace->putVariable(pair1, new Variable<T>(nullptr, nullptr, node->id(), 1));

            if (result->getScalar(0) == (T) 0.0f) {
                __flowPath->markBranch(node->id(), 0);
                __variableSpace->getVariable(pair0)->setNDArray(input->getNDArray());
                __variableSpace->getVariable(pair0)->markRemovable(false);
            } else {
                __flowPath->markBranch(node->id(),1);
                __variableSpace->getVariable(pair1)->setNDArray(input->getNDArray());
                __variableSpace->getVariable(pair1)->markRemovable(false);
            }

            return ND4J_STATUS_OK;
        };

        template class ND4J_EXPORT LogicSwitch<float>;
        template class ND4J_EXPORT LogicSwitch<float16>;
        template class ND4J_EXPORT LogicSwitch<double>;
    }
}
