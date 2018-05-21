//
// Created by raver119 on 16/11/17.
//

#include <graph/TimeHolder.h>

namespace nd4j {
    namespace graph {

        void TimeHolder::setOuterTime(int nodeId, Nd4jLong time) {
            _outer[nodeId] = time;
        }

        void TimeHolder::setInnerTime(int nodeId, Nd4jLong time) {
            _inner[nodeId] = time;
        }

        Nd4jLong TimeHolder::outerTime(int nodeId) {
            if (_outer.count(nodeId) == 0)
                return 0;

            return _outer[nodeId];
        }

        Nd4jLong TimeHolder::innerTime(int nodeId) {
            if (_inner.count(nodeId) == 0)
                return 0;

            return _inner[nodeId];
        }
    }
}
