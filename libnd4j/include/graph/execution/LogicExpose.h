//
// Created by raver119 on 12.11.2017.
//

#ifndef LIBND4J_LOGICEXPOSE_H
#define LIBND4J_LOGICEXPOSE_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        template <typename T>
        class LogicExpose {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}



#endif //LIBND4J_LOGICEXPOSE_H
