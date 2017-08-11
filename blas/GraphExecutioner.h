//
// @author raver119@gmail.com
//

#ifndef LIBND4J_GRAPHEXECUTIONER_H
#define LIBND4J_GRAPHEXECUTIONER_H

#include <graph/Graph.h>

namespace nd4j {
    namespace graph {
        class GraphExecutioner {
        public:
            /**
             * This method executes given Graph
             * @return
             */
            int execute(Graph *graph);
        };
    }
}

#endif //LIBND4J_GRAPHEXECUTIONER_H
