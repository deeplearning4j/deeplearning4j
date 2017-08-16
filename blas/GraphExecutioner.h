//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPHEXECUTIONER_H
#define LIBND4J_GRAPHEXECUTIONER_H

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <NDArray.h>
#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/Node.h>
#include <graph/Graph.h>


namespace nd4j {
    namespace graph {
        class GraphExecutioner {
        protected:

            Nd4jStatus static executeFlatNode(nd4j::graph::Graph *graph, nd4j::graph::Node *node, nd4j::graph::VariableSpace<float> *variableSpace);

        public:
            /**
             * This method executes given Graph
             * @return
             */
            Nd4jStatus static execute(nd4j::graph::Graph *graph);
        };
    }
}

#endif //LIBND4J_GRAPHEXECUTIONER_H
