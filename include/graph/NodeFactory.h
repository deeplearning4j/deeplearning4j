//
// Created by raver119 on 12.08.17.
//

#ifndef LIBND4J_NODEFACTORY_H
#define LIBND4J_NODEFACTORY_H

#include <graph/generated/node_generated.h>
#include <graph/Node.h>

namespace nd4j {
    namespace graph {
        class NodeFactory {

        public:
            static void * buildNode(const nd4j::graph::FlatNode *flatNode) {

                return nullptr;
            }
        };
    }
}

#endif //LIBND4J_NODEFACTORY_H
