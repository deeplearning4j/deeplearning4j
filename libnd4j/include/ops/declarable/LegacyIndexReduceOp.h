//
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYINDEXREDUCEOP_H
#define LIBND4J_LEGACYINDEXREDUCEOP_H

#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for IndexAccumulation operations. i.e. IndexMax or IndexAbsoluteMin etc
        *
        *   TODO: eventually we want this op class to return long long instead of T
        */
        template <typename T>
        class ND4J_EXPORT LegacyIndexReduceOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        public:
            LegacyIndexReduceOp();
            LegacyIndexReduceOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
            virtual LegacyOp<T>* clone();
        };
    }
}

#endif //LIBND4J_LEGACYINDEXREDUCEOP_H
