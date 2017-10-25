//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicExecutor.h>
#include <graph/execution/LogicScope.h>
#include <graph/execution/LogicWhile.h>
#include <graph/execution/LogicSwitch.h>
#include <graph/execution/LogicConditional.h>


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
            }
            return ND4J_STATUS_BAD_INPUT;
        }

        template class ND4J_EXPORT LogicExecutor<float>;
     //   template class ND4J_EXPORT LogicExecutor<float16>;
     //   template class ND4J_EXPORT LogicExecutor<double>;
    }
}