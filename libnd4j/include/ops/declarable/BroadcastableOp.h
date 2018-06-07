//
// Created by raver on 6/6/2018.
//

#ifndef LIBND4J_BROADCASTABLEOP_H
#define LIBND4J_BROADCASTABLEOP_H

#include <graph/Context.h>
#include "OpDescriptor.h"
#include "DeclarableOp.h"
#include "DeclarableCustomOp.h"

namespace nd4j {
    namespace ops {
        template <typename T>
        class ND4J_EXPORT BroadcastableOp : public DeclarableCustomOp<T>{
        protected:
            virtual Nd4jStatus validateAndExecute(Context<T> &block) = 0;
        public:
            BroadcastableOp(const char *name, int numTArgs, int numIArgs);
            ~BroadcastableOp();

            ShapeList *calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) override;
        };
    }
}


#endif //LIBND4J_BROADCASTABLEOP_H
