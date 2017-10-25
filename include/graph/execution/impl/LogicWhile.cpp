//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicWhile.h>
#include <GraphExecutioner.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicWhile<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();

            int scopeConditionIndex = node->input()->at(0).first;
            int scopeBodyIndex = node->input()->at(1).first;

            // we're running condition nodes now
            auto scope = graph->scopeById(scopeConditionIndex);
            int breaker = 0;
            while (true && breaker < 10000000) {
                int lastNode = 0;
                // we're running condition scope first
                for (auto v: *scope->nodes()) {
                    GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                    lastNode = v->id();
                }

                // now we should take result of the Scope run, and evaluate it
                auto result = __variableSpace->getVariable(lastNode)->getNDArray();

                if (Environment::getInstance()->isDebugAndVerbose())
                    result->printBuffer("Result of the last node:");

                // if result evaluates to 0.0 - condition returned FALSE
                if (result->getScalar(0) == (T) 0.0f)
                    break;
                else {
                    auto scopeBody = graph->scopeById(scopeBodyIndex);
                    int lastNode = 0;
                    for (auto v: *scopeBody->nodes()) {
                        GraphExecutioner<T>::executeFlatNode(graph, v, __variableSpace);
                        lastNode = v->id();
                    }
                }

                breaker++;
            }

            // if we've hit breaker limit - we should notify about that
            if (breaker >= 10000000) {
                nd4j_printf("While condition seems to be never ending, aborting...\n",  breaker);
                return ND4J_STATUS_KERNEL_FAILURE;
            }

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicWhile<float>;
//        template class ND4J_EXPORT LogicWhile<float16>;
//        template class ND4J_EXPORT LogicWhile<double>;
    }
}
