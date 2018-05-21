//
// Created by raver119 on 15.10.2017.
//

#ifndef LIBND4J_LOGICOP_H
#define LIBND4J_LOGICOP_H

#include "DeclarableOp.h"

namespace nd4j {
    namespace ops {

        /**
         * Logic ops are unique snowflakes in any Graph. They dramatically change Graph Execution process, by introducing loops, conditions, etc.
         *
         * Their code is the part of GraphExecutioner logic. But we still want them to be expressed via Graph
         * @tparam T
         */
        template <typename T>
        class ND4J_EXPORT LogicOp : public DeclarableOp<T> {
        protected:
            Nd4jStatus validateAndExecute(nd4j::graph::Context<T>& block) override;
        public:
            LogicOp(const char *name);
            ~LogicOp() = default;

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block) override;
        };
    }
}


#endif //LIBND4J_LOGICOP_H
