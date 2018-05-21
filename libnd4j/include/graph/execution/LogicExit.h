//
// Created by raver119 on 30.01.18.
//

#ifndef LIBND4J_LOGICEXIT_H
#define LIBND4J_LOGICEXIT_H

#include <pointercast.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class LogicExit {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}



#endif //LIBND4J_LOGICEXIT_H
