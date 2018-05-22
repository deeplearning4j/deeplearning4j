//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICWHILE_H
#define LIBND4J_LOGICWHILE_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class is responsible for execution logic of While logical abstraction
         *
         * Basic idea is simple: we take 2 scopes, one for condition and other one for body. and we re-execute body as long, as condition scope evaluates to TRUE
         * @tparam T
         */
        template <typename T>
        class LogicWhile {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICWHILE_H
