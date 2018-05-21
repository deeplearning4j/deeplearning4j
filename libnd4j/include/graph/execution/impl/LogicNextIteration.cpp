//
//  @author raver119@gmail.com
//

#include <graph/execution/LogicNextIteration.h>


namespace nd4j {
    namespace graph {
        template <typename T>
        Nd4jStatus LogicNextIeration<T>::processNode(Graph<T> *graph, Node<T> *node) {
            auto __variableSpace = graph->getVariableSpace();
            auto __flowPath = __variableSpace->flowPath();

            auto inputAddr = node->input()->at(0);

            auto var = __variableSpace->getVariable(inputAddr);

            Variable<T> *lvar = nullptr;
            if (__variableSpace->hasVariable(node->id(), 0))
                lvar = __variableSpace->getVariable(node->id(), 0);
            else
                lvar = new Variable<T>(nullptr, node->getName()->c_str(), node->id(), 0);

//            if (lvar->hasNDArray())
//                delete lvar->getNDArray();

            auto array = var->getNDArray();
            lvar->setNDArray(array);
            lvar->markReadOnly(true);

            return ND4J_STATUS_OK;
        }

        template class ND4J_EXPORT LogicNextIeration<float>;
        template class ND4J_EXPORT LogicNextIeration<float16>;
        template class ND4J_EXPORT LogicNextIeration<double>;
    }
}