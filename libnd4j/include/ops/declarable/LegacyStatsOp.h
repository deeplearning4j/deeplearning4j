//
// Created by raver119 on 17.10.2017.
//

#ifndef LIBND4J_LEGACYSTATSOP_H
#define LIBND4J_LEGACYSTATSOP_H

#include <ops/declarable/LegacyOp.h>

namespace nd4j {
    namespace ops {
        /**
        *   This class provides wrapper for SummaryStats operations: Variance and Standard Deviation
        */
        template <typename T>
        class ND4J_EXPORT LegacyStatsOp : public LegacyOp<T> {
        protected:
            Nd4jStatus validateAndExecute(Context<T>& block);
        public:
            LegacyStatsOp();
            LegacyStatsOp(int opNum);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
            virtual LegacyOp<T>* clone();
        };
    }
}


#endif //LIBND4J_LEGACYSTATSOP_H
