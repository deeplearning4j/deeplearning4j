//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICEXECUTOR_H
#define LIBND4J_LOGICEXECUTOR_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class acts as switch for picking logic execution based on opNum, unique for each logical op
         * @tparam T
         */
        template <typename T>
        class LogicExecutor {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}



#endif //LIBND4J_LOGICEXECUTOR_H
