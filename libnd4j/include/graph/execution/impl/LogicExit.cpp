//
//  @author raver119@gmail.com
//

#include <graph/execution/LogicExit.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicExit<T>::processNode(Graph<T> *graph, Node<T> *node) {
            // this op is basically no-op
            // we just know it exists

            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            Context<T> ctx(node->getContextPrototype(), __variableSpace);
            auto input = ctx.variable(0)->getNDArray();

            std::pair<int, int> pair0(node->id(), 0);

            if (!__variableSpace->hasVariable(pair0))
                __variableSpace->putVariable(pair0, new Variable<T>(nullptr, nullptr, node->id(), 0));

            __variableSpace->getVariable(pair0)->setNDArray(input);
            __variableSpace->getVariable(pair0)->markRemovable(false);

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicExit<float>;
        template class ND4J_EXPORT LogicExit<float16>;
        template class ND4J_EXPORT LogicExit<double>;
    }
}