//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicExecutor.h>
#include <graph/execution/LogicScope.h>
#include <graph/execution/LogicWhile.h>
#include <graph/execution/LogicSwitch.h>
#include <graph/execution/LogicConditional.h>
#include <graph/execution/LogicReturn.h>
#include <graph/execution/LogicExpose.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicExecutor<T>::processNode(Graph<T> *graph, Node<T> *node) {
            switch (node->opNum()) {
                case 0:
                    return LogicWhile<T>::processNode(graph, node);
                case 10:
                    return LogicScope<T>::processNode(graph, node);
                case 20:
                    return LogicConditional<T>::processNode(graph, node);
                case 30:
                    return LogicSwitch<T>::processNode(graph, node);
                case 40:
                    return LogicReturn<T>::processNode(graph, node);
                case 50:
                    return LogicExpose<T>::processNode(graph, node);
            }

            if (node->getName() == nullptr) {
                nd4j_printf("Unknown LogicOp used at node [%i]: [%i]\n", node->id(), node->opNum());
            } else {
                nd4j_printf("Unknown LogicOp used at node [%i:<%s>]: [%i]\n", node->id(), node->getName()->c_str(), node->opNum());
            }
            return ND4J_STATUS_BAD_INPUT;
        }

        template class ND4J_EXPORT LogicExecutor<float>;
        template class ND4J_EXPORT LogicExecutor<float16>;
        template class ND4J_EXPORT LogicExecutor<double>;
    }
}