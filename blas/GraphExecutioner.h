//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPHEXECUTIONER_H
#define LIBND4J_GRAPHEXECUTIONER_H

#include <graph/generated/node_generated.h>
#include <graph/generated/graph_generated.h>

#include <graph/Variable.h>
#include <graph/VariableSpace.h>
#include <graph/Node.h>
#include <graph/Graph.h>


namespace nd4j {
    namespace graph {
        class GraphExecutioner {
        protected:



        public:
            //static Nd4jStatus executeFlatNode(nd4j::graph::Graph *graph, nd4j::graph::Node *node, nd4j::graph::VariableSpace<float> *variableSpace);

            /**
             * This method executes given Graph
             * @return
             */
            static Nd4jStatus execute(nd4j::graph::Graph *graph);
        };
    }
}

#endif //LIBND4J_GRAPHEXECUTIONER_H
