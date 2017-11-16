//
// Created by raver119 on 16/11/17.
//

#include <graph/NodeState.h>

namespace nd4j {
    namespace graph {

        void NodeState::setInnerTime(Nd4jIndex time) {
            _inner = time;
        }

        void NodeState::setOuterTime(Nd4jIndex time) {
            _outer = time;
        }

        Nd4jIndex NodeState::innerTime() {
            return _inner;
        }

        Nd4jIndex NodeState::outerTime() {
            return _outer;
        }

        void NodeState::markActive(bool isActive) {
            _active = isActive;
        }

        bool NodeState::isActive() {
            return _active;
        }

        int NodeState::branch() {
            return _branch;
        }

        void NodeState::markBranch(int index) {
            _branch = index;
        }
    }
}