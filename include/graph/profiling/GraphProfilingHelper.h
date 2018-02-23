//
// Created by raver119 on 21.02.18.
//

#ifndef LIBND4J_GRAPHPROFILINGHELPER_H
#define LIBND4J_GRAPHPROFILINGHELPER_H


#include <graph/Graph.h>
#include "GraphProfile.h"

namespace nd4j {
    namespace graph {
        template <typename T>
        class GraphProfilingHelper {
        public:
            static GraphProfile* profile(Graph<T> *graph, int iterations);
        };
    }
}

#endif //LIBND4J_GRAPHPROFILINGHELPER_H
