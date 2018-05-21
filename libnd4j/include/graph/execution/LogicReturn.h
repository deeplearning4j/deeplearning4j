//
// Created by raver119 on 28.10.2017.
//

#ifndef LIBND4J_LOGICRETURN_H
#define LIBND4J_LOGICRETURN_H


#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class is responsible for execution logic of Return logical abstraction
         *
         * Basically we're just transferring input variable(s) to output variable(s), nothing beyond that
         * @tparam T
         */
        template <typename T>
        class LogicReturn {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}



#endif //LIBND4J_LOGICRETURN_H
