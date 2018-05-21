//
// Created by raver119 on 16/11/17.
//

#ifndef LIBND4J_NODESTATE_H
#define LIBND4J_NODESTATE_H

#include <pointercast.h>

namespace nd4j {
    namespace graph {
        class NodeState {
        private:
            // inner time spent on specific node
            Nd4jLong _inner = 0;

            // outer time spent on specific node
            Nd4jLong _outer = 0;
            
            // flag that shows if node is active or disabled (i.e. after Switch op)
            bool _active = true;

            bool _executed = false;

            // active divergence branch
            int _branch = 0;

            int _id = 0;
        public:
            NodeState(int id = 0);
            ~NodeState() = default;

            void setInnerTime(Nd4jLong time);
            void setOuterTime(Nd4jLong time);

            Nd4jLong innerTime();
            Nd4jLong outerTime();

            void markActive(bool isActive);
            bool isActive();

            int branch();
            void markBranch(int index);

            bool wasExecuted();
            void markExecuted(bool wasExecuted);
        };
    }
}


#endif //LIBND4J_NODESTATE_H
