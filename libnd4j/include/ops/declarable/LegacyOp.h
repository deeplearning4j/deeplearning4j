//
// Created by raver119 on 16.10.2017.
//

#ifndef LIBND4J_LEGACYOP_H
#define LIBND4J_LEGACYOP_H

#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {

        /**
        * This class is root abstraction for legacy XYZ ops wrappers.
        * All wrappers for specific op groups (i.e. LegacyTransformOp for Transform ops) are inheriting this class.
        *
        *
        */
        template <typename T>
        class LegacyOp : public DeclarableOp<T> {
        protected:
            // this field is mainly for debugging
            // it defines, which legacy op should be invoked on a given data
            int _opNum = -1;
            int _numInputs = 0;

            // All Op classes provide own specific implementation for this method
            virtual Nd4jStatus validateAndExecute(Context<T>& block) = 0;
        public:
            LegacyOp(int numInputs);
            LegacyOp(int numInputs, int opNum);

            // All Op classes provide own specific implementation for this method
            virtual ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block) = 0;
            virtual LegacyOp<T>* clone() = 0;
        };
    }
}


#endif //LIBND4J_LEGACYOP_H
