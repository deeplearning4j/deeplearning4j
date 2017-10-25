//
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYPAIRWISETRANSFORMOP_H
#define LIBND4J_LEGACYPAIRWISETRANSFORMOP_H

#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for Pairwise transform operations
        */
        template <typename T>
        class LegacyPairwiseTransformOp: public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Block<T>& block);
        public:
            LegacyPairwiseTransformOp();
            LegacyPairwiseTransformOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Block<T>& block);
        };
    }
}


#endif //LIBND4J_LEGACYPAIRWISETRANSFORMOP_H
