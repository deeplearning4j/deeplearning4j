//
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_FLOWPATH_H
#define LIBND4J_FLOWPATH_H

#include <map>
#include <pointercast.h>
#include <graph/NodeState.h>
#include <graph/FrameState.h>
#include <dll.h>

namespace nd4j {
    namespace graph {
        class ND4J_EXPORT FlowPath {
        private:
            std::map<int, NodeState> _states;
            std::map<Nd4jIndex, FrameState> _frames;

            void ensureNode(int nodeId);
            void ensureFrame(int nodeId);
        public:
            FlowPath() = default;
            ~FlowPath() = default;

            void setInnerTime(int nodeId, Nd4jIndex time);
            void setOuterTime(int nodeId, Nd4jIndex time);

            Nd4jIndex innerTime(int nodeId);
            Nd4jIndex outerTime(int nodeId);

            bool isNodeActive(int nodeId);
            void markNodeActive(int nodeId, bool isActive);

            bool wasExecuted(int nodeId);
            void markExecuted(int nodeId, bool wasExecuted);

            int branch(int nodeId);
            void markBranch(int nodeId, int index);

            // Frame-related methods

            void registerFrame(Nd4jIndex frameId);
            void forgetFrame(Nd4jIndex frameId);

            bool isFrameActive(Nd4jIndex frameId);
            void markFrameActive(Nd4jIndex frameId, bool isActive);

            bool isRewindPlanned(Nd4jIndex frameId);
            void planRewind(Nd4jIndex frameId, bool reallyRewind);

            int getRewindPosition(Nd4jIndex frameId);
            void setRewindPosition(Nd4jIndex frameId, int position);
            void setRewindPositionOnce(Nd4jIndex frameId, int position);

            void incrementNumberOfCycles(Nd4jIndex frameId);
            Nd4jIndex getNumberOfCycles(Nd4jIndex frameId);
        };
    }
}


#endif //LIBND4J_FLOWPATH_H
