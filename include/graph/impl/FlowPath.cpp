//
// Created by raver119 on 16/11/17.
//

#include <graph/FlowPath.h>

namespace nd4j {
    namespace graph {

        void FlowPath::ensureNode(int nodeId) {
            if (_states.count(nodeId) == 0) {
                NodeState state(nodeId);
                _states[nodeId] = state;
            }
        }

        void FlowPath::ensureFrame(int frameId) {
            if (_frames.count(frameId) == 0) {
                FrameState state(frameId);
                _frames[frameId] = state;
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

            return _states[nodeId].innerTime();
        }

        Nd4jIndex FlowPath::outerTime(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].outerTime();
        }

        bool FlowPath::isNodeActive(int nodeId) {
            ensureNode(nodeId);

            return _states[nodeId].isActive();
        }
            
        void FlowPath::markNodeActive(int nodeId, bool isActive) {
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

        bool FlowPath::isFrameActive(Nd4jIndex frameId) {
            ensureFrame(frameId);

            return _frames[frameId].wasActivated();
        }

        void FlowPath::markFrameActive(Nd4jIndex frameId, bool isActive) {
            ensureFrame(frameId);

            _frames[frameId].markActivated(isActive);
        }

        bool FlowPath::isRewindPlanned(Nd4jIndex frameId) {
            return _frames[frameId].isRewindPlanned();
        }

        void FlowPath::planRewind(Nd4jIndex frameId, bool reallyRewind) {
            _frames[frameId].planRewind(reallyRewind);
        }

        int FlowPath::getRewindPosition(Nd4jIndex frameId) {
            return _frames[frameId].getRewindPosition();
        }

        void FlowPath::setRewindPosition(Nd4jIndex frameId, int position) {
            _frames[frameId].setRewindPosition(position);
        }

        void FlowPath::setRewindPositionOnce(Nd4jIndex frameId, int position) {
            _frames[frameId].setRewindPositionOnce(position);
        }

        void FlowPath::registerFrame(Nd4jIndex frameId) {
            if (_frames.count(frameId) == 0)
                ensureFrame(frameId);
        }

        void FlowPath::forgetFrame(Nd4jIndex frameId) {
            if (_frames.count(frameId) > 0)
                _frames.erase(frameId);
        }

        void FlowPath::incrementNumberOfCycles(Nd4jIndex frameId) {
            _frames[frameId].incrementNumberOfCycles();
        }

        Nd4jIndex FlowPath::getNumberOfCycles(Nd4jIndex frameId) {
            return _frames[frameId].getNumberOfCycles();
        }


        bool FlowPath::wasExecuted(int nodeId) {
            return _states[nodeId].wasExecuted();
        }

        void FlowPath::markExecuted(int nodeId, bool wasExecuted) {
            _states[nodeId].markExecuted(wasExecuted);
        }

        GraphProfile* FlowPath::profile() {
            return &_profile;
        }
    }
}