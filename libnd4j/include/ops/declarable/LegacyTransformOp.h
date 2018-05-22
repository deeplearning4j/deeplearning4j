//
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYTRANSFORMOP_H
#define LIBND4J_LEGACYTRANSFORMOP_H


#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for Transform operations (i.e. Pow or OneMinus)
        */
        template <typename T>
        class ND4J_EXPORT LegacyTransformOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        public:
            LegacyTransformOp();
            LegacyTransformOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
            virtual LegacyOp<T>* clone();
        };
    }
}


#endif //LIBND4J_LEGACYTRANSFORMOP_H
