//
// Created by raver on 6/6/2018.
//

#include <op_boilerplate.h>
#include <pointercast.h>
#include <ops/declarable/BroadcastableOp.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        template <typename T>
        BroadcastableOp<T>::BroadcastableOp(const char *name, int numTArgs, int numIArgs) : DeclarableCustomOp<T>::DeclarableCustomOp(2, 1, name, false, numTArgs, numIArgs) {
            //
        }

        template <typename T>
        BroadcastableOp<T>::~BroadcastableOp() {
            // no-op
        }

        template <typename T>
        ShapeList *BroadcastableOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto shapeList = SHAPELIST();
            auto x = inputShape->at(0);
            auto y = inputShape->at(1);

            if (shape::equalsSoft(x, y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (shape::isScalar(x) && !shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(y, newshape);

                shapeList->push_back(newshape);
            } else if (!shape::isScalar(x) && shape::isScalar(y)) {
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            } else if (ShapeUtils<T>::areShapesBroadcastable(x, y)) {
                Nd4jLong *newshape = nullptr;
                ShapeUtils<T>::evalBroadcastShapeInfo(x, y, true, newshape, block.workspace());

                shapeList->push_back(newshape);
            } else {
                // in this case we'll throw exception later
                Nd4jLong *newshape;
                COPY_SHAPE(x, newshape);

                shapeList->push_back(newshape);
            }

            return shapeList;
        }

        template class ND4J_EXPORT BroadcastableOp<float>;
        template class ND4J_EXPORT BroadcastableOp<float16>;
        template class ND4J_EXPORT BroadcastableOp<double>;
    }
}