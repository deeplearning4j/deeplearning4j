//
// Created by raver119 on 16.10.2017.
//

#include "ops/declarable/LegacyPairwiseTransformOp.h"


namespace nd4j {
    namespace ops {
        template <typename T>
        LegacyPairwiseTransformOp<T>::LegacyPairwiseTransformOp() : LegacyOp<T>::LegacyOp(2) {
            // just a no-op
        }

        template <typename T>
        LegacyPairwiseTransformOp<T>::LegacyPairwiseTransformOp(int opNum) : LegacyOp<T>::LegacyOp(2, opNum) {
            // just a no-op
        }


        template <typename T>
        Nd4jStatus LegacyPairwiseTransformOp<T>::validateAndExecute(Context<T> &block) {
            auto x = INPUT_VARIABLE(0);
            auto y = INPUT_VARIABLE(1);
            auto z = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(x->isSameShape(y) || y->isScalar(), 0, "Node_%i: For Pairwise transforms shapes of both operands should be equal", block.getNodeId());

            int opNum = block.opNum() < 0 ? this->_opNum : block.opNum();

            NativeOpExcutioner<T>::execPairwiseTransform(opNum, x->getBuffer(), x->getShapeInfo(), y->getBuffer(), y->getShapeInfo(), z->getBuffer(), z->getShapeInfo(), block.getTArguments()->data());

            STORE_RESULT(*z);

            return ND4J_STATUS_OK;
        }

        /**
        *   Output shape of PWT operations always the same as input[0] shape, no exclusions.
        */
        template <typename T>
        ShapeList *LegacyPairwiseTransformOp<T>::calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) {
            auto inShape = inputShape->at(0);

            int *newShape;
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(inShape), int);
            memcpy(newShape, inShape, shape::shapeInfoByteLength(inShape));

            return new ShapeList(newShape);
        }

        template class ND4J_EXPORT LegacyPairwiseTransformOp<float>;
        template class ND4J_EXPORT LegacyPairwiseTransformOp<double>;
        template class ND4J_EXPORT LegacyPairwiseTransformOp<float16>;
    }
}