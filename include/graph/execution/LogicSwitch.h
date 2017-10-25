//
// Created by raver119 on 21.10.17.
//

#ifndef LIBND4J_LOGICSWITCH_H
#define LIBND4J_LOGICSWITCH_H

#include <pointercast.h>
#include <graph/Node.h>
#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class is responsible for execution logic of Switch logical abstraction
         *
         * It's ultra-simple. It does nothing, and can't be executed directly.
         *
         * @tparam T
         */
        template <typename T>
        class LogicSwitch {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICSWITCH_H
