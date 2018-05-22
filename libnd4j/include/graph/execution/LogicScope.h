//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICSCOPE_H
#define LIBND4J_LOGICSCOPE_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class is responsible for execution logic of Scope logical abstraction
         *
         * It's ultra-simple. It does nothing, and can't be executed directly.
         *
         * @tparam T
         */
        template <typename T>
        class LogicScope {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICSCOPE_H
