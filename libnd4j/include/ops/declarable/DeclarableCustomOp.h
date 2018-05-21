//
// Created by raver119 on 07.10.2017.
//

#ifndef LIBND4J_DECLARABLECUSTOMOP_H
#define LIBND4J_DECLARABLECUSTOMOP_H

#include <ops/declarable/DeclarableOp.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        class DeclarableCustomOp : public nd4j::ops::DeclarableOp<T> {
        protected:
            /**
             * This method executes this Op
             */
            virtual Nd4jStatus validateAndExecute(Context<T>& block) = 0;
        public:
            DeclarableCustomOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs);
            ~DeclarableCustomOp();

            virtual ShapeList* calculateOutputShape(ShapeList* inputShapes, nd4j::graph::Context<T>& block) = 0;
        };
    }
}

#endif //LIBND4J_DECLARABLECUSTOMOP_H
