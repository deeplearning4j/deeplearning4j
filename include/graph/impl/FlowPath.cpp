//
// Created by raver119 on 16/11/17.
//

#include <graph/FlowPath.h>

namespace nd4j {
    namespace graph {

        void FlowPath::ensureNode(int nodeId) {
            if (_states.count(nodeId) == 0) {
                NodeState state;
                _states[nodeId] = state;
            }
        }

        void FlowPath::setInnerTime(int nodeId, Nd4jIndex time) {
            ensureNode(nodeId);

            _states[nodeId].setInnerTime(time);
        }

        void FlowPath::setOuterTime(int nodeId, Nd4jIndex time) {
            ensureNode(nodeId);

            _states[nodeId].setOuterTime(time);
        }

        Nd4jIndex FlowPath::innerTime(int nodeId) {
            ensureNode(nodeId);

            _states[nodeId].innerTime();
        }

        Nd4jIndex FlowPath::outerTime(int nodeId) {
            ensureNode(nodeId);

            _states[nodeId].outerTime();
        }

        bool FlowPath::isActive(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].isActive();
        }
            
        void FlowPath::markActive(int nodeId, bool isActive) {
            ensureNode(nodeId);

            _states[nodeId].markActive(isActive);
        }

        int FlowPath::branch(int nodeId){
            ensureNode(nodeId);

            return _states[nodeId].branch();
        }

        void FlowPath::markBranch(int nodeId, int index) {
            ensureNode(nodeId);

            _states[nodeId].markBranch(index);
        }
    }
}