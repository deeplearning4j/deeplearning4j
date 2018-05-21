//
// Created by raver119 on 20.10.2017.
//

#ifndef LIBND4J_LOGICCONDITIONAL_H
#define LIBND4J_LOGICCONDITIONAL_H

#include <pointercast.h>
#include <Node.h>
#include <Graph.h>

namespace nd4j {
    namespace graph {
        /**
         * This class is responsible for execution logic of Conditional logical abstraction
         *
         * TL/DR: Class takes 2 ops/scopes with the same number of inputs/outputs and condtion.
         * Condition is evaluated, and based on its result - one of ops/scopes is executed.
         * Results of this execution will be copied to Conditional node, and every other op
         * in the graph will be sure that it's Conditional own result, both alternative nodes will
         * stay in disguise.
         *
         * @tparam T
         */
        template <typename T>
        class LogicConditional {
        public:
            static Nd4jStatus processNode(Graph<T>* graph, Node<T>* node);
        };
    }
}


#endif //LIBND4J_LOGICCONDITIONAL_H
