//
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_TIMEHOLDER_H
#define LIBND4J_TIMEHOLDER_H

#include <map>
#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class TimeHolder {
        private:
            std::map<int, Nd4jIndex> _outer;
            std::map<int, Nd4jIndex> _inner;


        public:

            TimeHolder() = default;
            ~TimeHolder() = default;


            void setOuterTime(int nodeId, Nd4jIndex time);
            void setInnerTime(int nodeId, Nd4jIndex time);


            Nd4jIndex outerTime(int nodeId);
            Nd4jIndex innerTime(int nodeId);
        };
    }
}

#endif //LIBND4J_TIMEHOLDER_H
