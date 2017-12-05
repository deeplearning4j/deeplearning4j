//
// Created by raver119 on 20.10.2017.
//

#include <graph/execution/LogicScope.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicScope<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // this op is basically no-op
            // we just know it exists
            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicScope<float>;
        template class ND4J_EXPORT LogicScope<float16>;
        template class ND4J_EXPORT LogicScope<double>;
    }
}