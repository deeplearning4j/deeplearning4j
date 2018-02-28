//
// Created by raver119 on 21.02.18.
//

#include <graph/profiling/GraphProfilingHelper.h>
#include <GraphExecutioner.h>

namespace nd4j {
    namespace graph {
        template<typename T>
        GraphProfile *GraphProfilingHelper<T>::profile(Graph<T> *graph, int iterations) {

            // saving original workspace
            auto varSpace = graph->getVariableSpace()->clone();

            // printing out graph structure
            graph->printOut();

            // warm up
            for (int e = 0; e < 1000; e++) {
                FlowPath fp;

                auto _vs = varSpace->clone();
                //_vs->workspace()->expandTo(100000);
                _vs->setFlowPath(&fp);
                GraphExecutioner<T>::execute(graph, _vs);

                delete _vs;
            }


            auto profile = new GraphProfile();
            for (int e = 0; e < iterations; e++) {
                FlowPath fp;

                // we're always starting from "fresh" varspace here
                auto _vs = varSpace->clone();
                //_vs->workspace()->expandTo(100000);
                _vs->setFlowPath(&fp);
                GraphExecutioner<T>::execute(graph, _vs);

                auto p = fp.profile();
                if (e == 0)
                    profile->assign(p);
                else
                    profile->merge(p);

                delete _vs;
            }

            delete varSpace;

            return profile;
        }


        template class GraphProfilingHelper<float>;
        template class GraphProfilingHelper<float16>;
        template class GraphProfilingHelper<double>;
    }
}
