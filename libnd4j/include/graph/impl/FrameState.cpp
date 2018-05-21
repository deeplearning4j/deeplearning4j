//
// Created by raver119 on 06.02.2018.
//

#include <graph/FrameState.h>


namespace nd4j {
    namespace graph {
        FrameState::FrameState(Nd4jLong id) {
            this->_id = id;
        }

        int FrameState::getNumberOfCycles() {
            return _numberOfCycles;
        }

        void FrameState::incrementNumberOfCycles() {
            ++_numberOfCycles;
        }

        bool FrameState::wasActivated() {
            return _activated;
        }

        void FrameState::markActivated(bool reallyActivated) {
            _activated = reallyActivated;
        }

        std::string &FrameState::getFrameName() {
            return _name;
        }

        bool FrameState::isRewindPlanned() {
            return _rewindPlanned;
        }

        int FrameState::getRewindPosition() {
            return _rewindPosition;
        }

        void FrameState::setRewindPosition(int pos) {
            _rewindPosition = pos;
        }

        void FrameState::setRewindPositionOnce(int pos) {
            if (_rewindPosition < 0)
                _rewindPosition = pos;
        }

        void FrameState::planRewind(bool reallyPlanning) {
            _rewindPlanned = reallyPlanning;
        }
    }
}