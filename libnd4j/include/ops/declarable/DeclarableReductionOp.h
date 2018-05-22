//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_DECLARABLE_REDUCTION_OP_H
#define LIBND4J_DECLARABLE_REDUCTION_OP_H

#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class ND4J_EXPORT DeclarableReductionOp : public nd4j::ops::DeclarableOp<T> {
        protected:
            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Context<T>& block) = 0;
        public:
            DeclarableReductionOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);
            ~DeclarableReductionOp();

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context<T>& block);
        };
    }
}

#endif //LIBND4J_DECLARABLE_REDUCTION_OP_H
