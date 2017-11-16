//
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_FLOWPATH_H
#define LIBND4J_FLOWPATH_H

#include <map>
#include <pointercast.h>
#include <graph/NodeState.h>

namespace nd4j {
    namespace graph {
        class FlowPath {
        private:
            std::map<int, NodeState> _states;

            void ensureNode(int nodeId);
        public:
            FlowPath() = default;
            ~FlowPath() = default;

            void setInnerTime(int nodeId, Nd4jIndex time);
            void setOuterTime(int nodeId, Nd4jIndex time);

            Nd4jIndex innerTime(int nodeId);
            Nd4jIndex outerTime(int nodeId);

            bool isActive(int nodeId);
            
            void markActive(int nodeId, bool isActive);

            int branch(int nodeId);
            void markBranch(int nodeId, int index);
        };
    }
}


#endif //LIBND4J_FLOWPATH_H
