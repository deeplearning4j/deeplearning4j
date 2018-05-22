//
// Created by raver119 on 17.10.2017.
//

#ifndef LIBND4J_LEGACYBROADCASTOP_H
#define LIBND4J_LEGACYBROADCASTOP_H

#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for broadcast operations. 
        */
        template <typename T>
        class ND4J_EXPORT LegacyBroadcastOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        public:
            LegacyBroadcastOp();
            LegacyBroadcastOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
            virtual LegacyOp<T>* clone();
        };
    }
}


#endif //LIBND4J_LEGACYBROADCASTOP_H
